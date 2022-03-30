from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

if __name__ == "__main__":
    ws = Workspace.from_config()
    datastore = ws.get_default_datastore()
    datastore.upload(src_dir='datasets/matterport/1LXtFkjw3qL',
                     target_path='datasets/matterport/1LXtFkjw3qL',
                     overwrite=True, show_progress=True)