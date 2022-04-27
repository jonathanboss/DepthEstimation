from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Dataset, Run
from azureml.core.authentication import InteractiveLoginAuthentication
import logging
import json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

dataset_paths = {
    "KITTI98": '/datasets/KITTI98/',
    "matterport": '/datasets/matterport/',
    "matterport_undistorted": '/datasets/matterport_undistorted2'
}

def load_setup():
    # Opening JSON file
    f = open('setup.json')

    # returns JSON object as a dictionary
    data = json.load(f)

    return data['dataset'], data['model']


def main(dataset_name, model):
    exp_name = model + '_' + dataset_name

    log.info(f"Running experiment in Azure ML using compute target")

    # Load workspace from 'config.json'
    workspace = Workspace.from_config()
    experiment = Experiment(
        workspace=workspace,
        name=exp_name
    )

    # Define dataset
    datastore = workspace.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, dataset_paths[dataset_name]))

    # Configure run of script
    config = ScriptRunConfig(
        source_directory='./src',
        script='train.py',
        compute_target='gpu-cluster',
        arguments=[
            '--data_path', dataset.as_named_input('input').as_mount(),
            '--model', model],

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


if __name__ == '__main__':
    dataset_name, model = load_setup()
    main(dataset_name, model)
