import requests

class WatsonX_Adapter:
    def __init__(self, api_url:str, api_key:str):
        self.base_url = api_url
        self.token = f'Bearer {api_key}'
    #
    def parse_response(self, res):
        return (res.status_code, res.text)
    #
    def list_dataset_files(self):
        res = requests.get(f'{self.base_url}files',
                           headers = {'Authorization': self.token})
        return self.parse_response(res)
    #
    def upload_dataset_file(self, json_file_path:str):
        res = requests.post(f'{self.base_url}files',
                            headers = {'Authorization': self.token},
                            files = {
                                'purpose': (None, 'tune'),
                                'file': open(json_file_path, 'rb')
                            })
        return self.parse_response(res)
    #
    def delete_dataset_file(self, fid:str):
        res = requests.delete(f'{self.base_url}files/{fid}',
                              headers = {'Authorization': self.token})
        return self.parse_response(res)
    #
    def create_tune(self, tune_name:str, training_file_ids:list,
                    task_id:str, # generation, classification, or summarization
                    validation_file_ids:list = list(),
                    # https://bam.res.ibm.com/docs/models
                    model_id:str = 'google/flan-t5-xl',
                    method_id:str = 'mpt', # Multitask Prompt Tuning
                    batch_size:int = 4, num_epochs:int = 12):
        res = requests.post(f'{self.base_url}tunes',
                            headers = {'Authorization': self.token,
                                       'Content-Type': 'application/json'},
                            json = {
                                'name': tune_name,
                                'model_id': model_id,
                                'task_id': task_id,
                                'method_id': method_id,
                                'training_file_ids': training_file_ids,
                                'validation_file_ids': validation_file_ids,
                                'parameters': {
                                    'batch_size': batch_size,
                                    'num_epochs': num_epochs
                                }})
        return self.parse_response(res)
    #
    def list_tunes(self):
        res = requests.get(f'{self.base_url}tunes',
                            headers = {'Authorization': self.token})
        return self.parse_response(res)
    #
    def delete_tunes(self, tune_id:str):
        res = requests.delete(f'{self.base_url}tunes/{tune_id}',
                              headers = {'Authorization': self.token})
        return self.parse_response(res)