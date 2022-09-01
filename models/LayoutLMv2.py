import re, random
import numpy as np

import torch
from transformers import LayoutLMv2Processor, LayoutLMv2ForQuestionAnswering
from PIL import Image
from utils import correct_alignment

from transformers.models.layoutlmv2.modeling_layoutlmv2 import LayoutLMv2Model
from transformers.models.layoutlmv2.processing_layoutlmv2 import LayoutLMv2Processor

class LayoutLMv2:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.processor = LayoutLMv2Processor.from_pretrained(config['model_weights'])
        self.model = LayoutLMv2ForQuestionAnswering.from_pretrained(config['model_weights'])
        self.page_retrieval = config['page_retrieval'].lower()
        self.ignore_index = 9999  # 0

    def forward(self, batch, return_pred_answer=False):

        question = batch['questions']
        context = batch['contexts']
        answers = batch['answers']

        if self.page_retrieval == 'logits':
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            for batch_idx in range(len(question)):
                images = [Image.open(img_path).convert("RGB") for img_path in batch['image_names'][batch_idx]]
                boxes = [(bbox * 1000).astype(int) for bbox in batch['boxes'][batch_idx]]
                document_encoding = self.processor(images, [question[batch_idx]] * len(images), batch["words"][batch_idx], boxes=boxes, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

                max_logits = -999999
                answer_page = None
                document_outputs = None
                for page_idx in range(len(document_encoding['input_ids'])):
                    # input_ids = document_encoding["input_ids"][page_idx].to(self.model.device)
                    # attention_mask = document_encoding["attention_mask"][page_idx].to(self.model.device)

                    page_inputs = {k: v[page_idx].unsqueeze(dim=0) for k, v in document_encoding.items()}
                    # Retrieval with logits is available only during inference and hence, the start and end indices are not used.
                    # start_pos = torch.LongTensor(start_idxs).to(self.model.device) if start_idxs else None
                    # end_pos = torch.LongTensor(end_idxs).to(self.model.device) if end_idxs else None

                    page_outputs = self.model(**page_inputs)

                    start_logits_cnf = [page_outputs.start_logits[batch_ix, max_start_logits_idx.item()].item() for batch_ix, max_start_logits_idx in enumerate(page_outputs.start_logits.argmax(-1))][0]
                    end_logits_cnf = [page_outputs.end_logits[batch_ix, max_end_logits_idx.item()].item() for batch_ix, max_end_logits_idx in enumerate(page_outputs.end_logits.argmax(-1))][0]
                    page_logits = np.mean([start_logits_cnf, end_logits_cnf])

                    if page_logits > max_logits:
                        answer_page = page_idx
                        document_outputs = page_outputs
                        max_logits = page_logits

                outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
                pred_answers.append(self.get_answer_from_model_output([document_encoding["input_ids"][answer_page]], document_outputs)[0] if return_pred_answer else None)
                pred_answer_pages.append(answer_page)

        else:

            if self.page_retrieval == 'oracle':
                images = [Image.open(img_path).convert("RGB") for img_path in batch['image_names']]

            elif self.page_retrieval == 'concat':

                images = []
                for batch_idx in range(len(batch['question_id'])):
                    images.append(self.get_concat_v_multi_resize([Image.open(img_path).convert("RGB") for img_path in batch['image_names'][batch_idx]]))  # Concatenate images vertically.

            boxes = [(bbox * 1000).astype(int) for bbox in batch['boxes']]
            encoding = self.processor(images, question, batch["words"], boxes=boxes, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

            start_pos, end_pos = self.get_start_end_idx(encoding, context, answers)
            outputs = self.model(**encoding, start_positions=None, end_positions=None)
            pred_answers = self.get_answer_from_model_output(encoding.input_ids, outputs) if return_pred_answer else None

            # print(pred_answers)
            for batch_idx in range(len(question)):
                if pred_answers[batch_idx] in batch['answers'][batch_idx]:
                    pred_start_pos = outputs.start_logits.argmax(-1)[batch_idx].item()
                    pred_end_pos = outputs.end_logits.argmax(-1)[batch_idx].item()
                    if pred_start_pos != start_pos[batch_idx]:
                        print("GT start pos {:} and pred start pos {:} are different!!!".format(start_pos[batch_idx], pred_start_pos))

                    if pred_end_pos != end_pos[batch_idx]:
                        print("GT end pos {:} and pred end pos {:} are different!!!".format(end_pos[batch_idx], pred_end_pos))

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

            elif self.page_retrieval == 'concat':
                pred_answer_pages = [batch['context_page_corresp'][batch_idx][pred_start_idx] if len(batch['context_page_corresp'][batch_idx]) > pred_start_idx else -1 for batch_idx, pred_start_idx in enumerate(outputs.start_logits.argmax(-1).tolist())]

        # if random.randint(0, 1000) == 0:
        #     print(batch['question_id'])
        #     for gt_answer, pred_answer in zip(answers, pred_answers):
        #         print(gt_answer, pred_answer)
        #
        #     for start_p, end_p, pred_start_p, pred_end_p in zip(start_pos, end_pos, outputs.start_logits.argmax(-1), outputs.end_logits.argmax(-1)):
        #         print("GT: {:d}-{:d} \t Pred: {:d}-{:d}".format(start_p.item(), end_p.item(), pred_start_p, pred_end_p))

        return outputs, pred_answers, pred_answer_pages

    def get_concat_v_multi_resize(self, im_list, resample=Image.BICUBIC):
        min_width = min(im.width for im in im_list)
        im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)), resample=resample) for im in im_list]

        # Fix equal height for all images (breaks the aspect ratio).
        heights = [im.height for im in im_list]
        im_list_resize = [im.resize((im.height, max(heights)), resample=resample) for im in im_list_resize]

        total_height = sum(im.height for im in im_list_resize)
        dst = Image.new('RGB', (min_width, total_height))
        pos_y = 0
        for im in im_list_resize:
            dst.paste(im, (0, pos_y))
            pos_y += im.height
        return dst

    def get_start_end_idx(self, encoding, context, answers):
        pos_idx = []

        for batch_idx in range(len(encoding.input_ids)):
            answer_pos = []
            for answer in answers[batch_idx]:
                answer_words_length = len(answer.split())

                for token_pos, token in enumerate(encoding.input_ids[batch_idx]):
                    if self.processor.tokenizer.decode(encoding.input_ids[batch_idx][token_pos:token_pos + answer_words_length]) == answer:
                        answer_pos.append([token_pos, token_pos + answer_words_length-1])

            if len(answer_pos) == 0:
                pos_idx.append([self.ignore_index, self.ignore_index])

            else:
                answer_pos = random.choice(answer_pos)  # To add variability, pick a random correct span.
                pos_idx.append(answer_pos)

        start_idxs = torch.LongTensor([idx[0] for idx in pos_idx]).to(self.model.device)
        end_idxs = torch.LongTensor([idx[1] for idx in pos_idx]).to(self.model.device)

        return start_idxs, end_idxs

    def get_answer_from_model_output(self, input_tokens, outputs):
        start_idxs = torch.argmax(outputs.start_logits, axis=1)
        end_idxs = torch.argmax(outputs.end_logits, axis=1)

        answers = [self.processor.tokenizer.decode(input_tokens[batch_idx][start_idxs[batch_idx]: end_idxs[batch_idx]+1]).strip() for batch_idx in range(len(input_tokens))]

        return answers