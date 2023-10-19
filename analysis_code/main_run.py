import subprocess



################ calc temporal discounting factor ###########
subprocess.run(["Rscript", "MCQ/main.R "])

################ plot group-level task behavior ############
subprocess.run(["python", "group_behavior.py"])


################ assess psychometric properties of survival curve characteristics ############
subprocess.run(["python", "task.py"])


################ assess psychometric properties of selfreport ############
subprocess.run(["python", "selfreport.py"])


################ assess psychometric properties of model parameters ############
subprocess.run(["python", "model_para.py"])

################ simulate model-gennerated data ############
subprocess.run(["python", "model_rep_calc.py"])

################ verify model predictions ############
subprocess.run(["python", "model_rep_plot.py"])

################ verify model predictions of all alternative models ############
subprocess.run(["python", "model_rep_multi_plot.py"]) # replace the figures in manuscript 

################ model comparison ############
subprocess.run(["python", "model_compare.py"]) 
subprocess.run(['matlab', "./matlab/modelComparision.m"])

################ correlation analysis ############
subprocess.run(["python", "simple_correlation_para.py"]) 
subprocess.run(["python", "simple_correlation_task.py"]) 

################# multi-dimensional scaling analyses ############
subprocess.run("python", 'EFA_preprocess.py')
subprocess.run(["Rscript", "EFA/igraph.R"]) 










