import subprocess
import sys

def run_command_live(config_path):
    """
    Executes a CLI command and prints its output live in real-time.

    Parameters:
    - command (str or list): The CLI command to execute. Can be a string or a list of arguments.
    """
    command = ['llamafactory-cli', 'train', config_path]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # Ensure output is handled as text, not bytes
        bufsize=1,  # Line-buffered
        universal_newlines=True
    )
    
    # Iterate over the output and error streams
    try:
        # Read output line by line
        for stdout_line in iter(process.stdout.readline, ''):
            print(stdout_line, end='')  # Print stdout line in real-time
        
        # Read error line by line
        for stderr_line in iter(process.stderr.readline, ''):
            print(stderr_line, end='', file=sys.stderr)  # Print stderr line in real-time
        
        process.stdout.close()
        process.stderr.close()
        
        return_code = process.wait()  # Wait for the process to complete
        return return_code
    
    except KeyboardInterrupt:
        process.terminate()  # Handle user interrupt
        print("Process terminated by user.")
        return -1
    
def train_llamafactory_cli(config_path):
    """
    Executes the 'llamafactory-cli train' command with the provided configuration file.

    Parameters:
    - config_path (str): Path to the configuration YAML file.

    Returns:
    - A tuple (stdout, stderr, returncode) where:
      - stdout: The standard output from the command.
      - stderr: The standard error from the command.
      - returncode: The return code of the command.
    """
    command = ['llamafactory-cli', 'train', config_path]
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True  # Raises CalledProcessError on non-zero return code
        )
        return (result.stdout, result.stderr, result.returncode)
    except subprocess.CalledProcessError as e:
        # Capture output and error if the command fails
        return (e.stdout, e.stderr, e.returncode)
    except Exception as e:
        # Handle other exceptions
        return (str(e), '', -1)

# Example usage
if __name__ == "__main__":
    checkpoint = 300
    for dataset_label in [
        "surdur_dergi",
        "surdur_gscholar1",
        "surdur_gscholar2",
        "surdur_gscholar3",
        "surdur_kitap1",
        "surdur_kitap2",
        "surdur_kitap3",
        ]:

        config_file = '/home/ubuntu/M3A/LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml'
        default_config = f"""
        ### model
        model_name_or_path: t3ai-org/pt-model

        ### method
        stage: sft
        do_train: true
        finetuning_type: lora
        lora_target: all

        ### dataset
        dataset: {dataset_label}
        template: llama3
        cutoff_len: 1024
        max_samples: 500
        max_length: 1024
        overwrite_cache: true
        preprocessing_num_workers: 16

        ### output
        output_dir: saves/llama3-8b/lora/sft
        logging_steps: 10
        save_steps: 90
        plot_loss: true
        overwrite_output_dir: false

        ### train
        per_device_train_batch_size: 4
        gradient_accumulation_steps: 2
        learning_rate: 1.0e-4
        num_train_epochs: {checkpoint/3+10}
        lr_scheduler_type: linear
        warmup_ratio: 0.1
        bf16: true
        ddp_timeout: 180000000
        resume_from_checkpoint: /home/ubuntu/M3A/LLaMA-Factory/saves/llama3-8b/lora/sft/checkpoint-{checkpoint}
        lora_alpha: 4
        lora_rank: 2

        ### eval
        val_size: 0.15
        per_device_eval_batch_size: 4
        eval_strategy: steps
        """

        with open(config_file, "w") as file:
            file.write(default_config)

        import time
        time.sleep(5)

        returncode = run_command_live(config_file)
        
        
        print("Return Code:")
        print(returncode)
        if returncode != 0:
            break

        checkpoint += 30
    