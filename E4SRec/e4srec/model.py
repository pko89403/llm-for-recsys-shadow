import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import (
    LlamaModel, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from loguru import logger

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


class LLM4Rec(nn.Module):
    def __init__(self, **args):
        super(LLM4Rec, self).__init__()
        self.args = args
        self.input_dim, self.output_dim = args["input_dim"], args["output_dim"]

        logger.info(f"Initializing language decoder ...")
        # add the lora module
        peft_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r=self.args["lora_r"],
            lora_alpha=self.args["lora_alpha"],
            lora_dropout=self.args["lora_dropout"],
            target_modules=self.args["lora_target_modules"],
            bias="none",
        )


        """
            https://discuss.huggingface.co/t/task-type-parameter-of-loraconfig/52879/4
            https://github.com/huggingface/peft/blob/v0.8.2/src/peft/utils/peft_types.py#L68-L73

            Overview of the supported task types:
            - SEQ_CLS: Text classification.
            - SEQ_2_SEQ_LM: Sequence-to-sequence language modeling.
            - Causal LM: Causal language modeling.
            - TOKEN_CLS: Token classification.
            - QUESTION_ANS: Question answering.
            - FEATURE_EXTRACTION: Feature extraction. Provides the hidden states which can be used as embeddings or features
            for downstream tasks.


            MODEL_TYPE_TO_PEFT_MODEL_MAPPING: dict[str, type[PeftModel]] = {
                "SEQ_CLS": PeftModelForSequenceClassification,
                "SEQ_2_SEQ_LM": PeftModelForSeq2SeqLM,
                "CAUSAL_LM": PeftModelForCausalLM,
                "TOKEN_CLS": PeftModelForTokenClassification,
                "QUESTION_ANS": PeftModelForQuestionAnswering,
                "FEATURE_EXTRACTION": PeftModelForFeatureExtraction,
            }

            PeftModelForFeatureExtraction의 forward 함수는 기본적으로 입력 데이터를 전처리하여 base_model에 전달하고 그 출력을 반환하는 구조입니다. 이를 통해 특정 작업 유형(예: "FEATURE_EXTRACTION")에 맞게 모델을 설정하고 학습하는 것을 가능하게 합니다. 구체적으로 살펴보면, 이 함수가 하는 작업은 다음과 같습니다:

            PeftConfig에 따라 입력 전처리:

            Prompt Learning 여부와 peft_type에 따라 입력 데이터를 전처리합니다.
            필요한 경우 Attention Mask를 업데이트하고, Position IDs 및 Token Type IDs를 무시합니다.
            입력을 base_model에 전달:

            전처리된 입력 데이터를 base_model에 전달합니다.
            따라서 "FEATURE_EXTRACTION" 작업 유형을 위해 특별히 추가된 로직은 없으며, 입력 데이터를 전처리하고 base_model에 전달하여 피처를 추출하는 방식으로 동작합니다. 이는 "FEATURE_EXTRACTION" 작업 자체가 입력 데이터로부터 임베딩 또는 피처를 추출하는 것이기 때문에, 별도의 추가 로직이 필요하지 않을 수 있습니다.

            하지만 모델의 기능을 완전히 이해하려면 전체 PeftModel 클래스와 관련된 설정, 전처리, 후처리 과정을 더 살펴봐야 합니다. PeftModelForFeatureExtraction 클래스의 동작은 다음과 같은 방식으로 이루어집니다:

            입력 전처리:

            입력 데이터(input_ids, attention_mask, inputs_embeds 등)를 받아서 필요에 따라 전처리합니다.
            Prompt Learning이 활성화된 경우, 프롬프트와 입력 임베딩을 결합합니다.
            base_model 호출:

            전처리된 입력 데이터를 base_model에 전달하여 피처를 추출합니다.
            base_model은 피처 추출을 위한 모델로, 일반적으로 트랜스포머 기반의 사전 학습된 언어 모델이 될 수 있습니다.
            출력 반환:

            base_model의 출력을 그대로 반환합니다.
            이 함수의 동작은 주로 입력 전처리에 집중되어 있으며, 모델의 설정에 따라 전처리 방식이 달라질 수 있습니다. 특이한 로직 없이 base_model에 입력을 전달하고 출력을 반환하는 구조는, 피처 추출 작업에서 단순히 임베딩을 얻는 것이 주 목적이기 때문입니다.

            다시 말해, "FEATURE_EXTRACTION" 작업 유형에서는 입력 데이터를 전처리하여 적절한 형식으로 base_model에 전달하고, base_model의 출력을 그대로 반환하는 것이 핵심입니다. 이 과정에서 특별한 추가 로직이 필요하지 않기 때문에, 현재의 함수 구현이 이를 잘 반영하고 있습니다.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args["base_model"],
            cache_dir=self.args["cache_dir"],
            device_map=self.args["device_map"]
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        self.model.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args["base_model"],
            use_fast=False,
            cache_dir=args["cache_dir"]
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.instruct_ids, self.instruct_mask = self.tokenizer(
            self.args["instruction_text"][0],
            truncation=True, padding=False,
            return_tensors="pt", add_special_tokens=False
        ).values()
        self.response_ids, self.response_mask = self.tokenizer(
            self.args["instruction_text"][1],
            truncation=True, padding=False,
            return_tensors="pt", add_special_tokens=False
        ).values()

        logger.info("Language decoder initialized.")

        self.task_type = args["task_type"]
        if self.task_type == "general":
            self.user_embeds = nn.Embedding.from_pretrained(
                self.args["user_embeds"],
                freeze=True
            )
            self.user_proj = nn.Linear(
                self.input_dim,
                self.model.config.hidden_size
            )
        self.input_embeds = nn.Embedding.from_pretrained(
            self.args["input_embeds"],
            freeze=True
        )
        self.input_proj = nn.Linear(
            self.input_dim,
            self.model.config.hidden_size
        )
        self.score = nn.Linear(self.model.config.hidden_size, self.output_dim, bias=False)

    def predict(self, inputs, inputs_mask):
        bs = inputs.shape[0]
        instruct_embeds = self.model.model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
        response_embeds = self.model.model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        if self.task_type == "general":
            # inputs [user, item1, item2, ... ]
            users = self.user_proj(self.user_embeds(inputs[:, 0].unsqueeze(1)))
            items = self.input_proj(self.input_embeds(inputs[:, 1:]))
            inputs = torch.cat([users, items], dim=1)
        else:
            inputs = self.input_proj(self.input_embeds(inputs))
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.model(inputs_embeds=inputs, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.last_hidden_state[:, -1]
        pooled_logits = self.score(pooled_output)

        return outputs, pooled_logits.view(-1, self.output_dim)

    def forward(self, inputs, inputs_mask, labels):
        outputs, pooled_logits = self.predict(inputs, inputs_mask)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pooled_logits, labels.view(-1))
            
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )