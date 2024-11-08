1. Q: Originality  
A: As highlighted in the introduction, we propose not only a normalization module to facilitate representation learning but also, **more importantly, exploring potential applications of the proposed 4D Occupancy World Model,** including controllable generation, continuous prediction, and e2e planning:
    - **Controllable Generation**: Users can specify expected action conditions to the world decoder for conditional cross-attention (Appendix B.1). This process generates various occupancy states, including **corner cases where high ego speed results in collisions** (Main Paper Fig. 5 and Appendix E.1).
    - **E2E Planning**: We devise a novel planner that uses an occupancy-based cost function as a reward (**not a loss function**) to select the optimal trajectory based on occ predictions, ensuring safe driving (Detailed in Response 6 to Reviewer 9BVs).
    - **Continuous Prediction**: When planning e2e, we use the predicted trajectory and BEV features at the current moment to forecast future states (occupancy and trajectory), enabling continuous prediction and planning.
2. Q: How ViDAR and DriveWorld perform planning task?  
A: They use images to predict LiDAR or occupancy to pretrain the BEV encoder, which is then used to **fine-tune UniAD for downstream planning**, focusing primarily on the pretraining.  
However, we integrate 4D occ forecasting, controllable generation, and e2e planning for **multi-task learning within a unified world model**.
3. Q: SoTA on other benchmarks?  
A: Cam4DOcc [CVPR24] first introduces the **4D occupancy forecasting benchmark** on OpenOccupancy; we outperform it (2.0\% gain in mIoU$_f$) and set a new SoTA.  
We conduct additional experiments on the **Lyft dataset, further demonstrating SoTA performance with 5.7% gain in $mIoU_f$ and 5.1% in $VPQ_f$ (See Table 2 in Response to Reviewer CPT6).** We will open source code to the public.
4. Q: Difference between ego status and action condition.  
A: 
    - Ego status reflects the vehicle's state at a specific time, including both **static and dynamic information** like position, pose, speed, etc.
    - Vehicle action refers to **movement trends** such as trajectory offset, acceleration, steering, etc.
    - Li et al. explored how ego status affects planning performance, while we propose using action conditions for controllable occ generation.
    - In Table 5, we compare the planner's performance w/ and w/o ego status, as Li suggested, for a fair comparison.