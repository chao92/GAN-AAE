# GAN
## File Structure
--AgentForEdgeGAN.py [main file for phase 1]
--Pridictor.py [main file for phase 2]
--Config.py [configuration file]
--Other code
--model [Location of the check points]  
--res [Location of the final pridiction output]  
--original_data  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--\*.txt.sdne [edge list of input graph]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--\*.txt.sdne_group [label of the vertices in the corresponding graph] *For validation 
## Usage
### Phase 1
python AgentForEdgeGAN.py ./original_data/[input file] [checkpoint name] [trainning rate]
### Phase 2
python Predictor.py ./original_data/[input file] [checkpoint name] ./res/[output file]
