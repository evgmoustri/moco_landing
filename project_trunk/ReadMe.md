### Thesis MOCO projects

## Files

1. OpenSim_Models : Contains the OpenSim musculoskeletal models

2. Data : Contains kinematic data from SCONE 

3. Results : Stores the analysis results
   

## Scripts

1. track_trunk_cases : track motion from scone with Gait2392.osim, sensitivity analysis for trunk flexion/extension, bending and rotation   
   - results for each case : project_trunk\Results\track_trunk 
   - results comparatively : project_trunk\Results\Compare\track_trunk

2. predict_trunk_cases : predict using motion from scone as guess with Gait2392.osim, sensitivity analysis for trunk flexion/extension, bending and rotation   
   - results for each case : project_trunk\Results\predict_trunk
   - results comparatively : project_trunk\Results\Compare\predict_trunk

3. predict_trunk_cases_flex20 : predict using motion from scone as guess with Gait2392.osim, sensitivity analysis for trunk , bending and rotation for trunk flexion 20  
   - results for each case : project_trunk\Results\predict_trunk_flex20 
   - results comparatively : project_trunk\Results\Compare\predict_trunk_flex20

4. predict_trunk_free: predict with relaxed lumbar DOFs
   - results for each case : project_trunk\Results\predict_trunk_free
   - results comparatively : project_trunk\Results\Compare\predict_trunk_free

 - extra functions: 
     1. compare_trunk_ben : compare and plot results from trunk bending cases, 
     2. compare_trunk_ext : compare and plot results from trunk flexion/extension cases
     3. compare_trunk_rot : compare and plot results from trunk rotation cases                                 
	      - input:                                       
	         1 for track_trunk_cases                                         
	         2 for predict_trunk_cases                                      
	         3 for predict_trunk_cases_flex20 
			 
	 
	 
# Authors and Contributors
- Authors: Evgenia Moustridi (evgmoustridi@gmail.com)
- Contributors: Konstantinos Risvas (krisvas@ece.upat.gr)


