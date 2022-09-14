import random, warnings
import numpy as np
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.nn import LayerNorm as BertLayerNorm

from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, ModelOutput, BaseModelOutput

import transformers.models.t5.modeling_t5


class LayoutT5Config(T5Config):
    def __init__(self, max_2d_position_embeddings=1024,  **kwargs):
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.hidden_dropout_prob = 0.1
        self.layer_norm_eps = 1e-12


class SpatialEmbeddings(nn.Module):
    """
    Spatial embedding by summing x, y, w, h projected by nn.Embedding to hidden size.
    """

    def __init__(self, config):
        super(SpatialEmbeddings, self).__init__()

        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.config = config

    def forward(self, bbox):
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])  # TODO Remove width and height to test how much important are they.
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])  # TODO Remove width and height to test how much important are they.

        embeddings = (
                left_position_embeddings
                + upper_position_embeddings
                + right_position_embeddings
                + lower_position_embeddings
                + h_position_embeddings
                + w_position_embeddings
        )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class HiLT5(T5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.spatial_embeddings = SpatialEmbeddings(config)
        self.emb_matcher = MLP(self.spatial_embeddings.config.hidden_size, 0, self.config.hidden_size, 1)

        self.page_tokens = config.page_tokens
        self.max_doc_pages = config.max_doc_pages

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        extra_kwargs_to_be_removed = ['bbox', 'attention_mask', 'num_pages']
        encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not any(argument.startswith(p) for p in irrelevant_prefix + extra_kwargs_to_be_removed)}

        # 2.2 replace input ids by the hierarchical layout-aware input embeddings
        page_embeddings = []
        for p_idx in range(max(model_kwargs['num_pages'])):
            textual_emb = self.shared(inputs_tensor[:, p_idx])  # read from default T5
            spatial_emb = self.emb_matcher(self.spatial_embeddings(model_kwargs['bbox'][:, p_idx]))
            inputs_embeds = textual_emb + spatial_emb

            encoder_outputs = encoder(
                input_ids=None,
                attention_mask=model_kwargs['attention_mask'][:, p_idx],
                inputs_embeds=inputs_embeds,
                **encoder_kwargs
            )

            hidden_states = encoder_outputs[0]
            page_embeddings.append(hidden_states[:, :self.page_tokens])

        document_embeddings = torch.cat(page_embeddings, dim=1)

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = None
        # model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        model_kwargs["encoder_outputs"]: ModelOutput = ModelOutput({'last_hidden_state': document_embeddings})
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "num_pages": kwargs.get('num_pages'),
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        num_pages=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> # training
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> # inference
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
            >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> # studies have shown that owning a dog is good for you.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            page_embeddings = []
            # for page_idx in range(self.max_doc_pages):
            for page_idx in range(max(num_pages)):
                textual_emb = self.shared(input_ids[:, page_idx])  # read from default T5
                spatial_emb = self.emb_matcher(self.spatial_embeddings(bbox[:, page_idx]))
                inputs_embeds = textual_emb + spatial_emb
                encoder_outputs = self.encoder(
                    input_ids=None,  # Input IDs must be None because input embeds is provided.
                    attention_mask=attention_mask[:, page_idx],
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                # Keep only [PAGE] token representation.
                hidden_states = encoder_outputs[0]
                page_embeddings.append(hidden_states[:, :self.page_tokens])

            document_embeddings = torch.cat(page_embeddings, dim=1)

            # attention_mask = torch.zeros([hidden_states.shape[0], self.num_doc_cls_tokens * self.doc_pages]).to(document_embeddings.device)  # Pages, hidden size. Make use of all information of the document embedding
            attention_mask = torch.zeros([hidden_states.shape[0], self.page_tokens * max(num_pages)]).to(document_embeddings.device)  # Pages, hidden size. Make use of all information of the document embedding
            for bs_idx in range(len(hidden_states)):
                attention_mask[bs_idx, :min(num_pages[bs_idx], self.max_doc_pages) * self.page_tokens] = 1

            attention_mask = attention_mask.to(document_embeddings.device)

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):  # EncoderOutputs is True when comes from _prepare_encoder_decoder_kwargs_for_generation, during .generation function.
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

            hidden_states = encoder_outputs[0]  # TODO - This should be replaced by document embeddings
            # TODO - Create the Multipage mask.

            """  ==== NEW ==== """
            # Without question in decoder.
            document_embeddings = hidden_states

            attention_mask = torch.zeros([hidden_states.shape[0], self.page_tokens * max(num_pages)])
            for bs_idx in range(len(hidden_states)):
                attention_mask[bs_idx, : min(num_pages[bs_idx], max(num_pages)) * self.page_tokens] = 1

            attention_mask = attention_mask.to(document_embeddings.device)
            """  ==== END NEW ==== """

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            # encoder_hidden_states=hidden_states,
            encoder_hidden_states=document_embeddings,  # Previous 'hidden states' in original T5
            encoder_attention_mask=attention_mask,  # Multi-page attention mask.
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class Proxy_HiLT5:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
        self.page_tokens = config.get('page_tokens', 10)
        self.max_doc_pages = config.get('max_pages', 1)

        config_x = LayoutT5Config.from_pretrained(config['model_weights'])
        config_x.page_tokens = self.page_tokens
        config_x.max_doc_pages = self.max_doc_pages
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])
        self.tokenizer.add_tokens("[PAGE]")  # Single representation
        # [self.tokenizer.add_tokens("[PAGE_{:d}]".format(p)) for p in range(self.num_doc_cls_tokens)]  # Different representation
        self.model = HiLT5.from_pretrained(config['model_weights'], config=config_x)


    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        context = batch['contexts']
        answers = batch['answers']
        num_pages = batch['num_pages']

        page_token_box = [0, 0, 1000, 1000]
        question_box = [0, 0, 1000, 1000]
        padding_box = [0, 0, 0, 0]
        eos_box = [0, 0, 0, 0]

        bs = len(context)
        if self.page_retrieval == 'logits':
            pass
            # outputs = []
            # pred_answers = []
            # pred_answer_pages = []
            # for batch_idx in range(bs):
            #     input_text = ["{:}: question: {:s}  context: {:s}".format("[PAGE]", q, c) for q, c in zip([question[batch_idx]]*len(context[batch_idx]), context[batch_idx])]
            #     tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
            #     boxes = torch.zeros([tokens.input_ids.shape[0], tokens.input_ids.shape[1], 4], dtype=torch.long)
            #     boxes[:] = torch.tensor(padding_box)
            #
            #     question_boxes = torch.tensor([question_box] * len(self.tokenizer("question: {:s}  context: ".format(question[batch_idx])).input_ids[:-1]))
            #     for page_idx in range(tokens.input_ids.shape[0]):
            #         if len(batch['words'][batch_idx][page_idx]) >= 1:
            #             context_boxes = torch.tensor(np.array([box for word, word_box in zip(batch['words'][batch_idx][page_idx], batch['boxes'][batch_idx][page_idx]) for box in [word_box] * len(self.tokenizer(word).input_ids[:-1])]))
            #             context_boxes = context_boxes[:self.tokenizer.model_max_length - len(question_boxes) - 1]  # Remove boxes out of model max length.
            #         else:
            #             context_boxes = torch.tensor(padding_box)
            #
            #         boxes[page_idx, :len(question_boxes)] = question_boxes
            #         boxes[page_idx, len(question_boxes): len(question_boxes) + len(context_boxes)] = context_boxes * 1000
            #         boxes[page_idx, len(question_boxes) + len(context_boxes)] = torch.tensor(eos_box)
            #
            #     boxes = boxes.to(self.model.device)
            #
            #     max_logits = -999999
            #     answer_page = None
            #     best_answer = None
            #     pred_answer, logits = self.get_answer_from_model_output(tokens, boxes)
            #     for p_ix in range(len(input_text)):
            #         if logits[p_ix] > max_logits:
            #             max_logits = logits[p_ix]
            #             answer_page = p_ix
            #             best_answer = pred_answer[p_ix]
            #
            #     outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
            #     # pred_answers.append(self.get_answer_from_model_output(document_outputs)[0] if return_pred_answer else None)
            #     pred_answers.append(best_answer)
            #     pred_answer_pages.append(answer_page)

        else:

            input_ids, attention_mask = [], []
            longest_sequence = 0

            """ TODO - Set the max sequence length to N(512/1024) and simplify this triplicated loop."""
            for batch_idx in range(bs):
                input_text = ["{:s}: question: {:s}  context: {:s}".format("[PAGE]" * self.page_tokens, question[batch_idx], c) for c in context[batch_idx]]
                tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
                input_ids.append(tokens.input_ids)
                attention_mask.append(tokens.attention_mask)
                longest_sequence = max(longest_sequence, tokens.input_ids.shape[-1])

            all_input_ids = torch.zeros([bs, max(num_pages), longest_sequence], dtype=torch.long)
            all_attention_masks = torch.zeros([bs, max(num_pages), longest_sequence], dtype=torch.long)
            for batch_idx in range(bs):
                all_input_ids[batch_idx, :num_pages[batch_idx], :input_ids[batch_idx].shape[-1]] = input_ids[batch_idx]
                all_attention_masks[batch_idx, :num_pages[batch_idx], :attention_mask[batch_idx].shape[-1]] = attention_mask[batch_idx]

            # boxes = []
            all_boxes = torch.zeros([bs, max(num_pages), longest_sequence, 4], dtype=torch.long)
            for batch_idx in range(bs):
                # document_boxes = torch.zeros([bs, num_pages[batch_idx], longest_sequence, 4], dtype=torch.long)
                # document_boxes = torch.zeros([num_pages[batch_idx], longest_sequence, 4], dtype=torch.long)
                # document_boxes[:] = torch.tensor(padding_box)

                question_boxes = torch.tensor([question_box] * len(self.tokenizer("question: {:s}  context: ".format(question[batch_idx])).input_ids[:-1]))

                for page_idx in range(num_pages[batch_idx]):
                    if len(batch['words'][batch_idx][page_idx]) >= 1:
                        context_boxes = torch.tensor(np.array([box for word, word_box in zip(batch['words'][batch_idx][page_idx], batch['boxes'][batch_idx][page_idx]) for box in [word_box]*len(self.tokenizer(word).input_ids[:-1])]))
                        context_boxes = context_boxes[:self.tokenizer.model_max_length - self.page_tokens - len(question_boxes) - 1]  # Remove boxes out of model max length.
                    else:
                        context_boxes = torch.tensor(padding_box)

                    """
                    document_boxes[page_idx, :self.page_tokens] = torch.tensor(page_token_box)
                    document_boxes[page_idx, self.page_tokens: self.page_tokens+len(question_boxes)] = question_boxes
                    document_boxes[page_idx, self.page_tokens + len(question_boxes): self.page_tokens+len(question_boxes)+len(context_boxes)] = context_boxes*1000

                    document_boxes[page_idx, self.page_tokens + len(question_boxes) + len(context_boxes)] = torch.tensor(eos_box)
                    """

                    all_boxes[batch_idx, page_idx, :self.page_tokens] = torch.tensor(page_token_box)
                    all_boxes[batch_idx, page_idx, self.page_tokens: self.page_tokens + len(question_boxes)] = question_boxes
                    all_boxes[batch_idx, page_idx, self.page_tokens + len(question_boxes): self.page_tokens + len(question_boxes) + len(context_boxes)] = context_boxes * 1000

                    all_boxes[batch_idx, page_idx, self.page_tokens + len(question_boxes) + len(context_boxes)] = torch.tensor(eos_box)

                # document_boxes = document_boxes.to(self.model.device)
                # boxes.append(document_boxes)

            all_input_ids = all_input_ids.to(self.model.device)
            all_boxes = all_boxes.to(self.model.device)
            all_attention_masks = all_attention_masks.to(self.model.device)
            # batch_idx = 0
            # page_idx = 0
            # print(len(input_ids[batch_idx][page_idx]), len(attention_mask[batch_idx][page_idx]), len(boxes[batch_idx][page_idx]))

            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)

            outputs = self.model(input_ids=all_input_ids, bbox=all_boxes, attention_mask=all_attention_masks, labels=labels, num_pages=num_pages)
            # outputs = self.model(input_ids=tokens.input_ids, bbox=boxes, attention_mask=tokens.attention_mask, labels=labels, num_pages=batch['num_pages'])
            # pred_answers = self.get_answer_from_model_output(outputs) if return_pred_answer else None
            pred_answers, logits = self.get_answer_from_model_output(all_input_ids, all_boxes, all_attention_masks, num_pages) if return_pred_answer else None

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

            else:
                pred_answer_pages = [-1 for _ in range(bs)]

        return outputs, pred_answers, pred_answer_pages

    def get_answer_from_model_output(self, input_ids, boxes, attention_mask, num_pages):
        bs = input_ids.shape[0]
        output = self.model.generate(input_ids=input_ids, bbox=boxes, attention_mask=attention_mask, num_pages=num_pages, output_scores=True, return_dict_in_generate=True)
        pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)

        """
        logits = np.zeros(len(output['scores'][0]))
        for seq_ix in range(len(output['scores'])):
            seq_logits = output['scores'][seq_ix].max(dim=-1)
            for batch_ix, token_id in enumerate(seq_logits.indices):
                logits[batch_ix] += seq_logits.values[batch_ix] if token_id not in [self.tokenizer.pad_token_id] else 0
        """

        all_logits = torch.stack(output.scores)
        best_logits = np.zeros(len(output['scores'][0]))
        for seq_ix in range(len(output['scores'])):
            for batch_ix in range(bs):
                token_id = output.sequences[batch_ix, seq_ix+1]
                best_logits[batch_ix] += all_logits[seq_ix, batch_ix, token_id] if token_id not in self.tokenizer.all_special_ids else 0

        return pred_answers, best_logits