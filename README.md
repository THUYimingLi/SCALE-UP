# SCALE-UP

# We first release our codes and part of models for demonstration. 
# We store the poisoned datasets and poisoned models for BadNets and WaNet in   https://www.dropbox.com/sh/lhgr6g8v7lohao2/AAArQpt5Vty3O0C4rdIr9s-ua?dl=0  and https://www.dropbox.com/sh/99cmqkqfcqpg555/AAAhyOSmP2tjJsRx0u3ViSLwa?dl=0 

#You can run python ./test.py to reimplement the results for WaNet

#To reimplement other results, you should first download the BadNets and WaNet folder from above links in ./ dictatory. Then you can use  torch_model_wrapper.py file to extract SPC scores for different poisoned models. The SPC scores will be stored in the saved_np/ file.
Then you can change the path in process("saved_np/WaNet/tiny_bd.npy") to test our approach for other attacks.  

#We will upload other datasets and poisoned images lately, and all poisoned samples are generated through BackdoorBox

