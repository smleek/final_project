https://www.sciencedirect.com/science/article/pii/S0950705115002737 

# Evidence and Interpretation

**Checkpoint:** November 19, 2026, by 11:59 PM Mountain Time  
**Team:** Brayden Humpherys, Sujin Leek, Alyssa Wilson  
**Route:** A
**Date:** 11/19/2026

Add a new, dated entry to the root `README.md` of your team's GitHub repository. Link the code, results, and figures or tables that support your summary. Keep earlier entries, commit the update, and submit the repository URL with that commit in Canvas. This file is optional supporting material; preliminary or negative results are welcome.

## Evidence so far

- **Route A:** Compare the new component with its reference under comparable conditions. What changed, and what does the evidence suggest?

We tried using MLSMOTE to account for the multiple labels found in our dataset. Wording revision: The idea was that MLSMOTE would use the multilabel structure when creating synthetic observations and could improve performance relative to oversampling methods that do not account for multiple labels.


We initially tried a version of MLSMOTE from GitHub. It did not fully align with the MLSMOTE paper by Charte et al. (2015): rather than creating separate neighbor bags for each minority label, `get_minority_instace()` grouped observations containing any tail label into one combined minority pool and then oversampled from that pool. In our comparison, standard MLSMOTE and Gaussian MLSMOTE both produced lower macro F1 and micro F1 than the baseline.

We then rewrote MLSMOTE so that each minority label would have its own neighbor bag rather than sharing one combined minority pool. This implementation in [mlsmote_updated.py](MLSMOTE/mlsmote_updated.py) is more closely aligned with Charte et al.'s Algorithms 1 and 2. The adapted version generated 282 synthetic observations and reduced MeanIR from 12.6823 to 8.3767, but it still scored lower than baseline: 0.257564 macro F1 versus 0.260124, and 0.519653 micro F1 versus 0.520970.

Oversampling changed the training counts for several labels, but the smallest minority labels remained at zero test F1. For example, Erotica increased from 40 to 85 training observations, Girls Love from 60 to 120, Boys Love from 89 to 202, and Gourmet from 93 to 186; each remained at 0.0000 test F1. Training F1 changed for several majority and less-extreme labels as well, including a 0.013795 improvement for Drama and a 0.032917 decline for Avant Garde.

The revised implementation increased macro precision slightly, from 0.397094 to 0.398497, while macro recall decreased from 0.240924 to 0.236890. Thus, predictions became slightly more conservative, rather than simply unchanged.


## Interpretation and next step

- Describe one informative error, failure, tradeoff, or limitation. How does it affect your conclusion?
- What did the evidence change in your thinking, and what will you test or revise next?

We are considering returning to the initial global-Gaussian oversampling approach separately from MLSMOTE because MLSMOTE has not improved predictive accuracy in our multilabel experiments. The notebook comparisons suggest that neighbor-based oversampling did not make the smallest minority genres learnable, so we will test whether representation changes, label thresholds, or class weighting have more effect than the oversampling design.
