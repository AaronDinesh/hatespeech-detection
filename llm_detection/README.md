# IMPORTANT
This project has been created to work with the uv python package manager. You can still use other python environment
managers like conda or venv, but it is up to you to make sure you install all the required packages. The rest of this
README will be instructions on how to run the files in this folder

# RCP AIaaS LLM Inference
First make sure that you have created a ```.env``` file in this folder and make sure the following environment variables
are defined:

1. OPENAI_API_KEY
2. WANDB_KEY
3. NTFY_TOPIC (optional if you want to use the ntfy service for push notifications)

The first file you will need to run is the ```process_inputs_multiprocessing.py``` file. This will accept paths to various directories and will process those inputs into a format that can be dispatched to the LLM endpoint. You will
need to edit the ```LLM_PROMPT``` variable in this file to define the prompt that will be sent to the LLM. The run the
file execute the following command 
```
uv run process_inputs_multiprocessing.py --data-dir <PATH_TO_MMHS150K_DIR> --output-dir <PATH_TO_OUTPUT_DIR>
```
Note \<PATH_TO_MMHS150K_DIR\> must be a path to the root of the MMHS150K dataset directory. The output directory must
also exist. This file with then save the LLM prompt and also save all the processes prompts in a ```prompts.jsonl.gz```.
This is a gzip compressed jsonl file (each line is a separate json object that contains the prompt and the image). Since
this file contains b64 encoded image data, the final prompts.jsonl.gz file can be about 6-7GB.

The next file you can run is the ```verify_processed_inputs.py``` file. This file will read the prompts.jsonl.gz that
was just generated and verify that the prompts are all in the correct format. You can run this by executing the following
command
```
uv run verify_processed_inputs.py --file-path <PATH_TO_JSONL_GZ_FILE>
```

Finally depending on what the prompt is, you will either run ```RCP_LLM_inference.py``` (this is when you want the LLM
to predict the exact hate label or give multiple guesses at the hate label) or ```RCP_LLM_inference.py``` (this is when
you want the LLM to output a hate score). For the ```RCP_LLM_inference.py``` file you can either get the LLM to say Hate
or NotHate labels or one of the 6 MMHS150K hate labels. This can be specified by commenting out the unneeded prompt
schema at the top of the file. Both files accept the same command line options so for brevity only one of them will be
listed here. The other file will be with the exact same commands. Run the file by executing the following command:
(First time run)
```
uv run RCP_LLM_inference_with_score.py --input-file <PATH_TO_JSONL_GZ_FILE> --output-dir <PATH_TO_OUTPUT_DIR> --model <RCP_MODEL_NAME>
```
(If restarting from a previously interrupted run)
```
uv run RCP_LLM_inference_with_score.py --input-file <PATH_TO_JSONL_GZ_FILE> --output-dir <PATH_TO_OUTPUT_DIR> --model <RCP_MODEL_NAME> --restart --results-file <PATH_TO_OUTPUT_RESULTS_JSONL_GZ_FILE>
```
Since this runs asynchronously, you can also specify the max number of concurrent async jobs as well as the number of
retries in case of error, but it is recommended to leave these values as is. Since there are ~150K prompts to process,
this may take several hours. A progress bar is printed to the terminal.

You might also want to run the ```verify_output.py``` file. This will read the output prompts.jsonl.gz file and
ensure that the LLM output follows a specific schema. It is up to you to create the correct validation schema using
pydantic for your task. Currently the file is setup to verify the output when the LLM is asked for a hatefulness score
from 0-4. This file can be run using:
(To check for errors)
```
uv run verify_outputs.py --file-path <PATH_TO_OUTPUT_RESULTS_JSONL_GZ_FILE> 
```
(To fix errors with occasional model re-prompt)
```
uv run verify_output.py --file-path <PATH_TO_OUTPUT_RESULTS_JSONL_GZ_FILE> --fix-errors --fixed-file-path <PATH_TO_STORE_THE_FIXED_FILE> --model <RCP_MODEL_NAME> --hateful-score --prompts-file <PATH_TO_PROMPTS_FILE>python 
```

Finally, you may want to compute the metrics for the LLM_output. This can be done using the ```check_annotations.py```
file in the ```app``` directory. 

The arguments for each file are well documented inside the file so consult them if unsure what each argument does.

# RCP LLM Fine-tuning
First make sure that you have the ```.env``` file and populate it as you did above. You then need to create the docker
image to use in run-ai. Make sure you edit the relevant parameters inside the .sh file. Then run the following command:
```
./build_docker_image.sh
```
Then push to the registry using the command:
```
docker push registry.rcp.epfl.ch/ee-559-<USERNAME>/<IMAGE_NAME>
```

Once the image is pushed then ssh onto the RCP Jumphost and run the following command to finetune the Llava model:
```
runai submit --image registry.rcp.epfl.ch/ee-559-dinesh/my-toolbox-llava:v0.3 --pvc home:/mnt/rcp -e HOME=${HOME} --pvc course-ee-559-scratch:/scratch --gpu 8 --node-pools default --large-shm --memory 32G --command -- python /mnt/rcp/hatespeech-detection/llm_detection/LlaVA_finetuning.py --dataset-json-path=/mnt/rcp/hatespeech-detection/data/MMHS150K/MMHS150K_GT.json --image-path=/mnt/rcp/hatespeech-detection/data/MMHS150K/img_resized/ --image-text-path=/mnt/rcp/hatespeech-detection/data/MMHS150K/img_txt/ --splits-path=/mnt/rcp/hatespeech-detection/data/MMHS150K/splits/ --model-path=/scratch/models/student_models/hf_models/llava-hf/llava-v1.6-mistral-7b-hf/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/52320fb52229c8d942b1dcb8b63b3dc8087bc83b/ --env-file=/mnt/rcp/hatespeech-detection/llm_detection/.env --model-save-path=/mnt/rcp/hatespeech-detection/llm_detection/lora_weights/ --checkpoint-save-path=/mnt/rcp/hatespeech-detection/llm_detection/checkpoints --resume-from=/mnt/rcp/hatespeech-detection/llm_detection/checkpoints/epochepoch\=03-maeval_mae\=0.8750.ckpt
```
The command above will try to resume from a checkpoint file that you give it and it saves one every epoch. The number of
GPUs can be any number you want, the fine-tuning script makes use of model parallelism to greatly speed up training when
compared with a single GPU. 

To run inference on the Llava model you can run the following command:
```
runai submit --image registry.rcp.epfl.ch/ee-559-dinesh/my-toolbox-llava:v0.3 --pvc home:/mnt/rcp -e HOME=${HOME} --pvc course-ee-559-scratch:/scratch --gpu 1 --node-pools default --large-shm --memory 32G --command -- python /mnt/rcp/hatespeech-detection/llm_detection/Llava_inference_Lightning_Inference.py --dataset-json-path=/mnt/rcp/hatespeech-detection/data/MMHS150K/MMHS150K_GT.json --image-path=/mnt/rcp/hatespeech-detection/data/MMHS150K/img_resized/ --image-text-path=/mnt/rcp/hatespeech-detection/data/MMHS150K/img_txt/ --splits-path=/mnt/rcp/hatespeech-detection/data/MMHS150K/splits/ --base-model-path=/scratch/models/student_models/hf_models/llava-hf/llava-v1.6-mistral-7b-hf/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/52320fb52229c8d942b1dcb8b63b3dc8087bc83b/ --adapter-path=/mnt/rcp/hatespeech-detection/llm_detection/lora_weights/3eph-unbalanced-dataset/ --output-metrics=/mnt/rcp/hatespeech-detection/llm_detection/balanced-llava.json --llm-output=/mnt/rcp/hatespeech-detection/llm_detection/llava-preds-balanced-e7.jsonl.gz --checkpoint-file=/mnt/rcp/hatespeech-detection/llm_detection/checkpoints/epochepoch\=07-maeval_mae\=0.7250.ckpt
```
This will load a particular checkpoint using the ```--checkpoint-file``` argument and then run inference. The output is
saved to the path specified. 