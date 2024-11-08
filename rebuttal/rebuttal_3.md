|semantic norm|bicycle|motorcycle|pedestrian|bus|truck|
|:-:|:-:|:-:|:-:|:-:|:-:|
|w/o|31.3|15.0|14.1|39.4|**33.3**|
|w/|**31.9**|**16.1**|**15.8**|**39.6**|**33.3**|

Table1. Ablation on semantic normalization with $\tilde{mIoU}_f$ on various classes.

1. Q: Motivate semantic normalization.  
A: We ablate on the semantic norm in Table1, and the results show that it especially **improves IoUs for small objects**. Besides, we encourage reviewers to **refer to our visualization results (Appendix Fig. 2)**, which effectively enhance the semantic discrimination of BEV features, alleviating the ray-shaped problem (Appendix D.2).

|Method|$IoU_c$|$IoU_f(0.8s)$|$\tilde{mIoU}_f$|$VPQ_f$|
|:-:|:-:|:-:|:-:|:-:|
|OpenOccupancy-C|14.0|13.5|13.7|-|
|SPC|1.42|-|-|-|
|PowerBEV-3D|26.2|24.5|25.1|27.4|
|Cam4DOcc|36.4|33.6|34.6|28.2|
|**Ours**|**40.6**|**39.3**|**40.0**|**33.3**|

Table2. Comparisons on Lyft dataset for forecasting Inflated Occupancy and Flow.

2. Q: SoTA on other benchmarks?  
A: We conduct additional experiments on the **Lyft dataset, further demonstrating SoTA performance with a 5.7% gain in $mIoU_f$ and 5.1% in $VPQ_f$.** This demonstrates the enhanced ability of our method to forecast future occupancy and model dynamic motions. We will open-source the code to the community for further research.

3. Q: How to avoid ego-status leakage to planner?  
A: When planning e2e, we use the predicted trajectory instead of the GT ego status as control conditions.  
    - **Utilizing the predicted trajectory** facilitates forecasting future occupancy based on the current action, **enabling continuous prediction and planning.**  
    - **In contrast, using GT actions may lead to the leakage of GT ego status to the planner**, as demonstrated by Li [CVPR24].  
    - Table 4 of Main paper demonstrates this point: We compare the planning performances between using GT and predicted trajectories, where the use of the GT one serves as the planning upper bound.

4. Q: Typo error of $S$.  
A: $S \in \mathbb{R}^{h \times w \times d \times 1}$ represents the predicted semantic label of each voxel after the argmax function. We will modify the statement. Thanks!  

5. Q: How are the semantic- and motion-conditional norm re-combined?  
A: The output features from two types of normalization are **concatenated and passed through MLPs for adaptive fusion** of semantic and dynamic information. We will add the relative details in the subsequent version. Thanks for your careful reading!