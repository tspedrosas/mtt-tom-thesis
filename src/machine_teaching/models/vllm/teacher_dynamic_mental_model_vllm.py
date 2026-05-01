#! /usr/bin/env python
from typing import Dict, List, Tuple, Union
from machine_teaching.models.vllm.teacher_static_mental_model_vllm import (
    TeacherStaticMentalModel,
)

class TeacherDynamicMentalModel(TeacherStaticMentalModel):
    """
    MM-Dynamic: identical prompting logic to the static version, but exposes
    `update_mental_model()` so that _ic_samples are recalibrated after every
    teaching round.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # unpack the initial demonstrations coming from round-0 simulation
        if isinstance(self._ic_samples, Tuple):
            self._pre_ic, self._post_ic = list(self._ic_samples[0]), list(self._ic_samples[1])
        else:                             # (degenerate case – only one list)
            self._pre_ic, self._post_ic = list(self._ic_samples), []

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #
    def update_mental_model(
        self,
        new_pre: List[Dict],
        new_post: List[Dict],
    ) -> None:
        """
        `new_pre`  : examples (x, student-pred-pre) that *preceded* an explanation
        `new_post` : the same inputs together with teacher explanation & student
                     post-predictions.
        We simply append and rebuild the tuple consumed by the parent class.
        """
        self._pre_ic.extend(new_pre)
        self._post_ic.extend(new_post)
        self._ic_samples = (self._pre_ic, self._post_ic)   # what TeacherStaticMM expects
