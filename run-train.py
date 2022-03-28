from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Dataset, Run
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if __name__ == '__main__':
    log.info(f"Running experiment in Azure ML using compute target")

    # Load workspace from 'config.json'
    workspace = Workspace.from_config()
    experiment = Experiment(
        workspace=workspace,
        name='dataset-test'
    )

    # Define dataset
    datastore = workspace.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, '/datasets/KITTI98/'))

    # Configure run of script
    config = ScriptRunConfig(
        source_directory='./src',
        script='train.py',
        compute_target='gpu-cluster',
        arguments=[
            '--data_path', dataset.as_named_input('input').as_mount()],
    )

    # load Conda environment from local config file
    env = Environment.from_conda_specification(
        name='pytorch-env',
        file_path="./pytorch-env.yml",
    )
    # Specify a GPU base image
    env.docker.enabled = True
    env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04'
    config.run_config.environment = env

    # Submit the run to AzureMl
    run = experiment.submit(config)
    aml_url_handle = run.get_portal_url()
    print('Printing Azure ML experiment run handle.')
    print(aml_url_handle)
    run.wait_for_completion(show_output=True)
