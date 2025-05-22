#    Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" finetuning vision–language task (ERNIE‑ViL)
Updated for **PaddlePaddle 2.x** static graph.  All legacy 1.x Fluid APIs that
were removed (e.g. `py_reader`, `fluid.initializer.TruncatedNormal`) have been
replaced with their modern equivalents.  The script still uses the static
executor/parallel‑executor path so that the surrounding training code remains
unchanged.
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import time
import datetime
import json
import argparse
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
#  Paddle 2.x ‑ static‑graph setup
# ──────────────────────────────────────────────────────────────────────────────
import paddle                        # main namespace (ops / functional)
import paddle.nn as nn               # layers / initialisers
import paddle.nn.functional as F     # functional API
from paddle.fluid.io import DataLoader  # static replacement for py_reader
import paddle.fluid as fluid         # keep for Program/Executor compatibility

paddle.enable_static()               # we build a static computation graph

# ──────────────────────────────────────────────────────────────────────────────
#  Project‑specific imports (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
from reader.vcr_finetuning import VCRDataJointReader
from reader.hm_finetuning import HMDataReader
from model.ernie_vil import ErnieVilModel, ErnieVilConfig
from optim.optimization import optimization
from utils.args import print_arguments
from utils.init import init_checkpoint, init_pretraining_params
from utils.pandas_scripts import clean_data, double_data
from args.finetune_args import parser
from sklearn.metrics import roc_auc_score

args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
READERS = {"vcr": VCRDataJointReader, "hm": HMDataReader}
MODELS  = {"vcr": "create_vcr_model", "hm": "create_vcr_model"}

# ──────────────────────────────────────────────────────────────────────────────
#  Helper: build placeholders + DataLoader (replacement for py_reader)
# ──────────────────────────────────────────────────────────────────────────────

def _build_dataloader(pyreader_name: str, shapes, dtypes, capacity=30):
    """Return (feed_list, dataloader) ready for static graph."""
    feed_list = []
    for idx, (shape, dtype) in enumerate(zip(shapes, dtypes)):
        shp = [None if s == -1 else s for s in (shape or [1])]
        if dtype == "float":
            dtype = "float32"
        feed_list.append(
            paddle.static.data(name=f"{pyreader_name}_f{idx}", shape=shp, dtype=dtype)
        )

    dataloader = DataLoader.from_generator(
        feed_list         = feed_list,
        capacity          = capacity,
        iterable          = False,
        use_double_buffer = False,
    )
    return feed_list, dataloader

# ──────────────────────────────────────────────────────────────────────────────
#  Model‑builder (HM / VCR share the same architecture here)
# ──────────────────────────────────────────────────────────────────────────────

def create_vcr_model(pyreader_name, ernie_config, task_group, is_prediction=False):
    # ---------------- input spec ----------------
    shapes = [
        [-1, args.max_seq_len, 1],   # src_id
        [-1, args.max_seq_len, 1],   # pos_id
        [-1, args.max_seq_len, 1],   # sent_id
        [-1, args.max_seq_len, 1],   # task_id
        [-1, args.max_seq_len, 1],   # input_mask
        [-1, args.max_img_len, args.feature_size],  # image_embedding
        [-1, args.max_img_len, 5],  # image_loc
        [-1, args.max_img_len, 1],  # image_mask
        [-1, 1],                    # labels
        [-1, 1],                    # qids
        [],                         # task_index (scalar)
        [-1, 1],                    # binary_labels
    ]
    dtypes = [
        'int64','int64','int64','int64','float32',
        'float32','float32','float32',
        'int64','int64','int64','float32'
    ]
    for _ in task_group:
        shapes.append([])
        dtypes.append('float32')

    feed_list, pyreader = _build_dataloader(pyreader_name, shapes, dtypes)
    inputs = feed_list  # for static graph, inputs are the placeholders

    # unpack (keep the original order)
    (src_ids, pos_ids, sent_ids, task_ids, input_mask,
     image_embeddings, image_loc, image_mask,
     labels, q_ids, task_index, binary_labels, *extra_task_weights) = inputs

    # ---------------- backbone ----------------
    ernie_vil = ErnieVilModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        image_embeddings=image_embeddings,
        image_loc=image_loc,
        input_image_mask=image_mask,
        config=ernie_config,
    )
    h_cls, h_img = ernie_vil.get_pooled_output()

    task_conf = task_group[0]
    fusion_fea = ernie_vil.get_match_score(
        text=h_cls,
        image=h_img,
        dropout_rate=task_conf["dropout_rate"],
        mode=task_conf["fusion_method"],
    )

    # ---------------- head & loss ----------------
    if is_prediction:
        num_choice = int(task_conf['num_choice'])
        task_name  = task_conf.get('task_prefix', 'vcr')

        score = paddle.static.nn.fc(
            x           = fusion_fea,
            size        = 1,
            weight_attr = paddle.ParamAttr(
                name        = task_name + "_fc.w_0",
                initializer = nn.initializer.TruncatedNormal(std=0.02),
            ),
            bias_attr   = paddle.ParamAttr(name = task_name + "_fc.b_0"),
        )
        score = paddle.reshape(score, shape=[-1, num_choice])
        loss, softmax = F.softmax_with_cross_entropy(score, labels, return_softmax=True)
        acc  = paddle.static.accuracy(softmax, labels)
        pred = paddle.argmax(score, axis=1)

        mean_loss = paddle.mean(loss)
        task_vars = [mean_loss, acc, pred, q_ids, labels, score]
        for v in task_vars:
            v.persistable = True
        return pyreader, task_vars

    # ---------- training (possibly multi‑task weighted) ----------
    mean_loss = paddle.zeros([1], dtype='float32')
    mean_acc  = paddle.zeros([1], dtype='float32')
    start_idx = 0
    for task_conf in task_group:
        task_weight = extra_task_weights[start_idx]
        start_idx  += 1
        task_name   = task_conf.get('task_prefix', 'vcr')

        score = paddle.static.nn.fc(
            x           = fusion_fea,
            size        = 1,
            weight_attr = paddle.ParamAttr(
                name        = task_name + "_fc.w_0",
                initializer = nn.initializer.TruncatedNormal(std=0.02),
            ),
            bias_attr   = paddle.ParamAttr(name = task_name + "_fc.b_0"),
        )
        loss = F.binary_cross_entropy_with_logits(score, binary_labels)
        soft = F.softmax(paddle.reshape(score, [-1, int(task_conf['num_choice'])]), axis=-1)
        acc  = paddle.static.accuracy(soft, labels)
        mean_loss += paddle.mean(loss) * task_weight
        mean_acc  += acc * task_weight

    task_vars = [mean_loss, mean_acc, score, binary_labels]
    for v in task_vars:
        v.persistable = True
    return pyreader, task_vars

#MODELS = {"vcr": create_vcr_model, "vqa": create_vqa_model, "refcoco+": create_refcoco_model}
MODELS = {"vcr": create_vcr_model, "hm": create_vcr_model}

def predict_wrapper(args,
                    exe,
                    ernie_config,
                    task_group,
                    test_prog=None,
                    pyreader=None,
                    graph_vars=None):
    """Context to do validation.
    """
    reader_name = READERS[args.task_name]

    # Allow inputting the name of any
    split = args.split
    #if args.do_val:
    ##    split="val"
    #elif args.do_test:
    #    split="test"

    data_reader = reader_name(
        task_group,
        split=split,
        vocab_path=args.vocab_path,
        is_test=True,
        shuffle=False,
        batch_size=args.batch_size,
        epoch=args.epoch)

    if args.do_test:
        assert args.init_checkpoint is not None, "[FATAL] Please use --init_checkpoint '/path/to/checkpoints' \
                                                  to specify you pretrained model checkpoints"

        init_pretraining_params(exe, args.init_checkpoint, test_prog)
        print(("testing on %s %s split") % (args.task_name, args.test_split))

    def predict(exe=exe, pyreader=pyreader):
        """
            inference for downstream tasks
        """
        # pyreader.decorate_tensor_provider(data_reader.data_generator())
        pyreader.set_batch_generator(
        data_reader.data_generator(), places=[exe.place])
        pyreader.start()

        cost = 0
        appear_step = 0
        task_acc = {}
        task_steps = {}
        steps = 0
        case_f1 = 0
        appear_f1 = 0
        time_begin = time.time()
        task_name_list = [v.name for v in graph_vars]
        fetch_list = task_name_list

        print('task name list : ', task_name_list)
        sum_acc = 0
        res_arr = []

        # Val, rcac
        label_list, pred_list = [], []
        quesid2ans, quesid2prob = {}, {}

        # For some reason, themodel always omits the last 52 elements in our json files
        # It is not the fault of the data generator who spits them out -- No idea where the problem could be
        # For simplicity we just tag them on once more, creating longer jsonls, but reading in everything
        while True:
            try:
                outputs = exe.run(fetch_list=fetch_list, program=test_prog)
                each_acc = outputs[1][0]
                preds = np.reshape(outputs[2], [-1])
                qids = np.reshape(outputs[3], [-1])
                labels = np.reshape(outputs[4], [-1])
                #scores = np.reshape(outputs[5], [-1, 4]
                scores = np.reshape(outputs[5], [-1])

                label_list.extend(labels)
                pred_list.extend(scores)
                
                for qid, l in zip(qids, preds):
                    quesid2ans[qid] = l

                for qid, l in zip(qids, scores):
                    quesid2prob[qid] = l


                sum_acc += each_acc
                steps += 1
        
                if steps % 10 == 0:
                    print('cur_step:', steps, 'cur_acc:', sum_acc / steps)
                #format_result(res_arr, qids.tolist(), preds.tolist(), labels.tolist(), scores.tolist())
            except fluid.core.EOFException:
                print("EXCEPTING")
                pyreader.reset()
                break

        used_time = time.time() - time_begin

        #with open(args.result_file, "w") as f:
        #    for r in res_arr:
        #        f.write(r + "\n")


        # Dump preds to csv for submission
        dump_csv(quesid2ans, quesid2prob, "./data/hm/" + args.split + args.exp + ".csv")

        print("average_acc:", sum_acc / steps)
        print("rocauc:", roc_auc_score(label_list, pred_list))
        ret = {}
        ret["acc"] = "acc: %f" % (sum_acc / steps)  
        for item in ret:
            try:
                ret[item] = ret[item].split(':')[-1]
            except:
                pass
        return ret
    return predict


def dump_csv(quesid2ans: dict, quesid2prob: dict, path):

    print("LEN:", len(quesid2ans), len(quesid2prob))

    d = {"id": [int(tensor) for tensor in quesid2ans.keys()], "proba": list(quesid2prob.values()), 
        "label": list(quesid2ans.values())}
    results = pd.DataFrame(data=d)
    
    print(results.info())

    results.to_csv(path_or_buf=path, index=False)


def get_optimizer(total_loss, train_program, startup_prog, args):
    """
        optimization func
    """
    decay_steps_str=args.decay_steps
    if decay_steps_str == "":
        decay_steps = []
    else:
        decay_steps = [int(s) for s in decay_steps_str.split(";")]
    scheduled_lr = optimization(
         loss=total_loss,
         warmup_steps=args.warmup_steps,
         num_train_steps=args.num_train_steps,
         learning_rate=args.learning_rate,
         train_program=train_program,
         startup_prog=startup_prog,
         weight_decay=args.weight_decay,
         scheduler=args.lr_scheduler,
         decay_steps=decay_steps,
         lr_decay_ratio=args.lr_decay_ratio)
    return scheduled_lr


def main(args):
    """
       Main func for downstream tasks
    """
    print("finetuning tasks start")
    ernie_config = ErnieVilConfig(args.ernie_config_path)
    ernie_config.print_config()

    with open(args.task_group_json) as f:
        task_group = json.load(f)
        print('task: ', task_group)

    startup_prog = fluid.Program()
    if args.do_train and args.do_test:
        print("can not set both do_train and do_test as True")
        return 

    model_name = MODELS[args.task_name]
    if args.do_train:
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, model_outputs = model_name(
                    pyreader_name='train_reader', ernie_config=ernie_config, task_group=task_group)

                total_loss = model_outputs[0]
                scheduled_lr = get_optimizer(total_loss, train_program, startup_prog, args)


    if args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, model_outputs  = model_name(
                    pyreader_name='test_reader', ernie_config=ernie_config, task_group=task_group, is_prediction=True)
                total_loss = model_outputs[0]

        test_prog = test_prog.clone(for_test=True)
    
    if args.use_gpu:
        gpu_id = 0
        if os.getenv("FLAGS_selected_gpus"):
            gpu_id = int(os.getenv("FLAGS_selected_gpus"))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()

    print("theoretical memory usage: ")
    if args.do_train:
        print(fluid.contrib.memory_usage(
             program=train_program, batch_size=args.batch_size))
    if args.do_test:
        print(fluid.contrib.memory_usage(
            program=test_prog, batch_size=args.batch_size))

    nccl2_num_trainers = 1
    nccl2_trainer_id = 0
    print("args.is_distributed:", args.is_distributed)
    trainer_id = 0
    if args.is_distributed:
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        worker_endpoints_env = os.getenv("PADDLE_TRAINER_ENDPOINTS")
        current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
        worker_endpoints = worker_endpoints_env.split(",")
        trainers_num = len(worker_endpoints)

        print("worker_endpoints:{} trainers_num:{} current_endpoint:{} \
              trainer_id:{}".format(worker_endpoints, trainers_num,
                                    current_endpoint, trainer_id))

        # prepare nccl2 env.
        config = fluid.DistributeTranspilerConfig()
        config.mode = "nccl2"
        if args.nccl_comm_num > 1:
            config.nccl_comm_num = args.nccl_comm_num
        if args.use_hierarchical_allreduce and trainers_num > args.hierarchical_allreduce_inter_nranks:
            config.use_hierarchical_allreduce=args.use_hierarchical_allreduce
            config.hierarchical_allreduce_inter_nranks=args.hierarchical_allreduce_inter_nranks

            assert config.hierarchical_allreduce_inter_nranks > 1
            assert trainers_num % config.hierarchical_allreduce_inter_nranks == 0

            config.hierarchical_allreduce_exter_nranks = \
                trainers_num / config.hierarchical_allreduce_inter_nranks

        t = fluid.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id,
            trainers=worker_endpoints_env,
            current_endpoint=current_endpoint,
            program=train_program,
            startup_program=startup_prog)

        nccl2_num_trainers = trainers_num
        nccl2_trainer_id = trainer_id

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint and args.init_checkpoint != "":
            sys.stderr.write('############################WARNING############################')
            sys.stderr.write('####### using init_pretraining_params, not init_checkpoint ####')
            sys.stderr.write('## meaning hyper param e.g. lr won\'t inherit from checkpoint##')
            sys.stderr.write('###############################################################')
            init_pretraining_params(exe, args.init_checkpoint, train_program)

        reader_name=READERS[args.task_name]
        data_reader = reader_name(
            task_group,
            split=args.split,
            vocab_path=args.vocab_path,
            batch_size=args.batch_size,
            epoch=args.epoch,)

    exec_strategy = fluid.ExecutionStrategy()
    if args.use_fast_executor:
        exec_strategy.use_experimental_executor = True
    exec_strategy.num_threads = 2
    
    exec_strategy.num_iteration_per_drop_scope = min(10, args.skip_steps)

    build_strategy = fluid.compiler.BuildStrategy()
    build_strategy.fuse_all_reduce_ops = False

    if args.use_fuse:
        build_strategy.fuse_all_reduce_ops = True

    if args.do_train:
        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=total_loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
            main_program=train_program,
            num_trainers=nccl2_num_trainers,
            trainer_id=nccl2_trainer_id)

    if args.do_test: 
        predict = predict_wrapper(
            args,
            exe,
            ernie_config,
            task_group,
            test_prog=test_prog,
            pyreader=test_pyreader,
            graph_vars=model_outputs)
        result = predict()

    if args.do_train:
        # train_pyreader.decorate_tensor_provider(data_reader.data_generator())
        train_pyreader.set_batch_generator(
            data_reader.data_generator(), places=[place])
        train_pyreader.start()

        # For testing purposes
        preds = []
        targets = []

        steps = 0
        time_begin = time.time()
        node_nums = 1 #int(os.getenv("PADDLE_NODES_NUM"))
        used_time_all = 0 
        while steps < args.num_train_steps:
            try:
                steps += node_nums
                skip_steps = args.skip_steps * node_nums
                fetch_list = []
                if nccl2_trainer_id == 0 and steps % skip_steps == 0:
                    task_name_list = [v.name for v in model_outputs]
                    fetch_list = task_name_list
                    fetch_list.append(scheduled_lr.name)
                
                time_begin = time.time()
                outputs = train_exe.run(fetch_list=fetch_list)
                if outputs:
                    print("feed_queue size", train_pyreader.queue.size())
                    progress_file = data_reader.get_progress()
                    epoch = progress_file["current_epoch"]
                    current_file_index = progress_file["current_file_index"]
                    total_file =  progress_file["total_file"]
                    current_file = progress_file["current_file"]
                    print(
                        "epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                        "acc: %f"
                        % (epoch, current_file_index, total_file, steps,
                           outputs[0][0],
                           outputs[1][0]))
                    print("steps:", steps)
                    print("save_steps:", args.save_steps)

                    # For Validation & testing purposes
                    preds.append(outputs[2][0])
                    targets.append(outputs[3][0])

                    if steps % 500 == 0:
                        print("Train-RCAC", roc_auc_score(targets, preds))
                        preds = []
                        targets = []

                    np_lr = outputs[-1:]

                    date_str = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")

                    np_lr = float(np.mean(np_lr[0]))
                    print("%s current learning_rate:%.8f" % (date_str, np_lr))

                    if steps % args.save_steps == 0:
                        save_path = os.path.join(args.checkpoints, "step_" + str(steps) + str(args.split))
                        print("save_path:", save_path)
                        fluid.io.save_persistables(exe, save_path, train_program)
                    time_end = time.time()
                    used_time = time_end - time_begin
                    time_end = time_begin
                    print("used_time:", used_time)  

                if steps == args.stop_steps:
                    break

            except fluid.core.EOFException:
                train_pyreader.reset()
                break


if __name__ == '__main__':
    print_arguments(args)

    if args.task_name == "hm":
        # Create pretrain.jsonl & traindev data
        # clean_data("./data/hm")
        # This handles formatting for the E-Models. There needs to be a label column & some data needs to be copied to the end for length requirements.
        double_data("./data/hm")

    main(args)
