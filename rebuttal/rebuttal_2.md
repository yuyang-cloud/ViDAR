1. Q: Define $o,a,s$.  
A: In our method, $o$ represents image observations, $a$ are ego actions, and $s$ denotes future states (occupancy and dynamic flow). We will add this statement in the paper.  
2. Q: Typo error of $S$.  
A: $S \in R^{h * w * d * 1}$ represents the predicted semantic label after the argmax function. We will modify the statement.  

|velocity|acceleration|$mIoU_c$|$mIoU_f(1s)$|$\tilde{mIoU}_f$|$VPQ_f^*$
|:-:|:-:|:-:|:-:|:-:|:-:
|||28.7|26.4|26.8|33.5
|√||28.9|27.5|27.8|33.9
||√|29.0|27.8|28.0|34.1
|√|√|**29.2**|**28.0**|**28.2**|**34.5**

Table1. Ablation on vel and acc.

3. Q: Take acceleration as action?  
A: We conducted an ablation using velocity and acceleration as action conditions in Table1. It show that **acceleration improves forecasting performance better**, as it provides the changing trend of velocity over various timestamps. Thanks for your advice!  
4. Q: Not clear about motion conditional normalization and $(\gamma, \beta)$.  
A: We account for two types of motions:
    - **Ego motion**: The ego pose is encoded into $(\gamma^e, \beta^e)$ for ego motion-aware norm.
    - **Other agents' motion**: We use 3D flow prediction $\mathcal{F} \in R^{h * w * d * 3}$, which indicates the voxel-wise dynamic flow, and encode it into $(\gamma^f, \beta^f)$ for fine-grained motion-aware norm.
    - Then, we **normalize the BEV features** by $\gamma * F^{bev} + \beta$ (Main Paper Fig.3). The norm parameters, derived from semantic or motion predictions, effectively **transform the feature distribution by integrating semantic and dynamic information**.
5. Q: Using predicted trajectory during train or test?  
A: When planning e2e, we utilize the predicted trajectories for **both training and testing**. This not only **prevents the GT ego action from leaking to the planner** but also **facilitates model learning with predicted trajectories**, as you mentioned, enabling improved performance during testing.  
6. Q: Details of the cost function in the planner.  
A:  We will add the relative details:
    - **Agent- and Road-Safety Costs:** We extract the occupied grids from occ predictions, forming a 2D cost map that penalizes trajectory candidates that overlap with agents or fall outside the road.
    - **Learned Cost Volume:** We use a learnable head based on the BEV feature to generate another 2D cost map, providing a more comprehensive evaluation of the complex environment.
    - The above costs are summed to select the optimal trajectory.