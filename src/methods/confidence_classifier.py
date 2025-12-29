"""Confidence classification for knowledge selection.

This module classifies the confidence level of knowledge selection into three categories:
- Confident: High confidence in the top-1 knowledge
- Undecided: Uncertain between top-2 or top-3 candidates
- Unclear: Low overall confidence
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class ConfidenceClassifier:

    def __init__(
        self,
        conf_sum_prob: float = 0.8,
        conf_top1_ratio: float = 0.8,
        unconf_sum_prob: float = 0.65,
        sum_any_kn_prob: float = 0.6,
        second_first_diff_ratio: float = 0.3,
        third_second_diff_ratio: float = 0.5,
        top_k_num: int = 3,
    ):
        """Initialize the confidence classifier.

        Args:
            conf_sum_prob: Top-K sum probability threshold for Confident class
            conf_top1_ratio: Top-1 ratio in Top-K for Confident class
            unconf_sum_prob: Top-K sum probability threshold for Undecided class
            sum_any_kn_prob: Sum probability threshold for Unclear class
            second_first_diff_ratio: Second/First ratio threshold for Confident
            third_second_diff_ratio: Third/Second ratio threshold for Undecided
            top_k_num: Number of top candidates to consider (default: 3)
        """
        self.conf_sum_prob = conf_sum_prob
        self.conf_top1_ratio = conf_top1_ratio
        self.unconf_sum_prob = unconf_sum_prob
        self.sum_any_kn_prob = sum_any_kn_prob
        self.second_first_diff_ratio = second_first_diff_ratio
        self.third_second_diff_ratio = third_second_diff_ratio
        self.top_k_num = top_k_num

    def classify(
        self,
        knowledge_scores: List[float],
        knowledge_candidates: List[int],
        trivia_scores: Optional[List[float]] = None,
    ) -> Tuple[List[int], str]:
        if len(knowledge_scores) < 2:
            return [0], "Confident"

        # Use knowledge_scores directly (matching original implementation)
        knowledge_score_list = knowledge_scores
        scores_tensor = torch.tensor(knowledge_score_list)
        classification_prob = F.softmax(scores_tensor, dim=-1)

        idx_rank_list = self._get_sorted_indices(knowledge_score_list)
        top_k_idx_list = idx_rank_list[: self.top_k_num]
        tmp_topk_prob_list = [classification_prob[idx].item() for idx in top_k_idx_list]

        topk_sum_prob = sum(tmp_topk_prob_list)
        top_prob = tmp_topk_prob_list[0]
        top1_ratio_of3 = top_prob / topk_sum_prob if topk_sum_prob > 0 else 0.0
        second_prob = tmp_topk_prob_list[1] if len(tmp_topk_prob_list) > 1 else 0.0
        second_diff_ratio = second_prob / top_prob if top_prob > 0 else 0.0

        # Use trivia scores directly (matching original implementation)
        kn_trivia_scores = trivia_scores
        if trivia_scores is not None:
            if len(trivia_scores) != len(knowledge_score_list):
                raise ValueError(
                    f"Trivia scores length mismatch: {len(trivia_scores)} != {len(knowledge_score_list)}"
                )

        if topk_sum_prob >= self.conf_sum_prob and top1_ratio_of3 >= self.conf_top1_ratio:
            kn_select_idx_list = [int(np.argmax(knowledge_score_list))]
            class_label = "Confident"
        elif second_diff_ratio <= self.second_first_diff_ratio:
            kn_select_idx_list = [int(np.argmax(knowledge_score_list))]
            class_label = "Confident"

        elif topk_sum_prob >= self.unconf_sum_prob:
            third_prob = tmp_topk_prob_list[2] if len(tmp_topk_prob_list) > 2 else 0.0
            third_diff_ratio = third_prob / second_prob if second_prob > 0 else 0.0

            if third_diff_ratio <= self.third_second_diff_ratio:
                tmp_kn_idx_list = top_k_idx_list[:2]
            else:
                tmp_kn_idx_list = top_k_idx_list

            if kn_trivia_scores is not None:
                topk_kn_triviascore_list = [kn_trivia_scores[idx] for idx in tmp_kn_idx_list]
                if -1 in topk_kn_triviascore_list:
                    raise ValueError("Trivia score contains invalid value (-1)")
                kn_select_idx_list = [
                    k_idx
                    for _, k_idx in sorted(
                        zip(topk_kn_triviascore_list, tmp_kn_idx_list), key=lambda x: -x[0]
                    )
                ]
            else:
                kn_select_idx_list = tmp_kn_idx_list
            class_label = "Undecided"

        else:
            any_kn_prob_list = []
            any_kn_idx_list = []
            for idx in idx_rank_list:
                any_kn_prob_list.append(classification_prob[idx].item())
                any_kn_idx_list.append(idx)
                if sum(any_kn_prob_list) >= self.sum_any_kn_prob:
                    break

            if kn_trivia_scores is not None:
                any_kn_triviascore_list = [kn_trivia_scores[idx] for idx in any_kn_idx_list]
                if -1 in any_kn_triviascore_list:
                    raise ValueError("Trivia score contains invalid value (-1)")
                kn_select_idx_list = [
                    k_idx
                    for _, k_idx in sorted(
                        zip(any_kn_triviascore_list, any_kn_idx_list), key=lambda x: -x[0]
                    )
                ]
            else:
                kn_select_idx_list = any_kn_idx_list
            class_label = "Unclear"

        return kn_select_idx_list, class_label

    @staticmethod
    def _get_sorted_indices(scores: List[float]) -> List[int]:
        return sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
