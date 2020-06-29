from kaggle_runner.kernels.kernel import KaggleKernelOnlyPredict


from transformers import *

class ToxicPredictModel(KaggleKernelOnlyPredict):
    def __init__(self, model_path):
        super(ToxicPredictModel, self).__init__(model_path)
        self.only_predict = True
        self.model_path = model_path

    def save_model(self, output_dir="./models/"):
        model = self.model

        from transformers import WEIGHTS_NAME, CONFIG_NAME
# Step 1: Save a model, configuration and vocabulary that you have fine-tuned

# If we have a distributed model, save only the encapsulated model
# (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
        model_to_save = model.module if hasattr(model, 'module') else model

# If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_pretrained(output_dir)

    def build_and_set_model(self):
        """load pretrained one"""
        # Example for a Bert model
        #config = BertConfig.from_json_file(output_config_file)
        #model = BertForQuestionAnswering(config)
        #state_dict = torch.load(output_model_file)
        #model.load_state_dict(state_dict)
        #tokenizer = BertTokenizer(output_vocab_file, do_lower_case=args.do_lower_case)
        model = BertForQuestionAnswering.from_pretrained(output_dir)
        tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed

    def prepare_train_dev_data(self):
        pass

    def prepare_test_data(self, data_config=None):
        pass

    def check_predict_details(self):
        pass

    def peek_data(self):
        pass

def only_predict():
    pass

def test_init():
    k = ToxicPredictModel(".")
    assert k is not None
    k.save_model()

import torch

def test_load():
    output_model_file='/kaggle/input/bert-for-toxic-classfication-trained/2020-06-21_XLMRobertaModel_tpu_trained.bin'
    state_dict = torch.load(output_model_file)
    k.model.load_state_dict(state_dict)

    print(model)
