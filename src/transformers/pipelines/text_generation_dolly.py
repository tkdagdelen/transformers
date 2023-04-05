from collections.abc import Sequence
import enum
import warnings
import numpy as np

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, TF_MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.utils import add_end_docstrings, is_tf_available
from transformers.pipelines.base import PIPELINE_INIT_ARGS, Pipeline


if is_tf_available():
    import tensorflow as tf


class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


# @add_end_docstrings(PIPELINE_INIT_ARGS)
class TextGenerationDollyPipeline(Pipeline):
    """
    Language generation pipeline using any `ModelWithLMHead`. This pipeline predicts the words that will follow a
    specified text prompt.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def _sanitize_parameters(
        self,
        *args,
        **generate_kwargs,):
        return {}, {}, {}

    def __call__(self, text_inputs, **kwargs):
        preprocessed = self.preprocess(text_inputs,**kwargs)
        response = self._forward(preprocessed, **kwargs)
        return self.postprocess(response, **kwargs)       

    def preprocess(self, instruction, do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **generate_kwargs):
        PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
        input_ids = self.tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to("cuda")
        return input_ids

    def _forward(self, input_ids, 
                        do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs):
      # each of these is encoded to a single token
      response_key_token_id = self.tokenizer.encode("### Response:")[0]
      end_key_token_id = self.tokenizer.encode("### End")[0]

      return self.model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                  do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)[0].cpu()

    def postprocess(self, gen_tokens, do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs):
      response_key_token_id = self.tokenizer.encode("### Response:")[0]
      end_key_token_id = self.tokenizer.encode("### End")[0]
      # find where the response begins
      response_positions = np.where(gen_tokens == response_key_token_id)[0]

      if len(response_positions) >= 0:
          response_pos = response_positions[0]
          
          # find where the response ends
          end_pos = None
          end_positions = np.where(gen_tokens == end_key_token_id)[0]
          if len(end_positions) > 0:
              end_pos = end_positions[0]

          return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos], ).strip()
      return None
       
