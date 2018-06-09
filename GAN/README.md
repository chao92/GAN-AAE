# GAN
## File Structure
--AgentForEdgeGAN.py [main file for step 1]  
--Pridictor.py [main file for step 2]  
--Config.py [configuration file]  
--Other code  
**--model** [Location of the check points]  
**--res** [Location of the final pridiction output]  
**--original_data**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--\*.txt.data [edge list of input graph]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--\*.txt.data_group [label of the vertices in the corresponding graph] *For validation 
## Usage
### Step 1
python3 AgentForEdgeGAN.py ./original_data/[input file] [checkpoint name] [trainning rate]
### Step 2
python3 Predictor.py ./original_data/[input file] [checkpoint name] ./res/[output file]
