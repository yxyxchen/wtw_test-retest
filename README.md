## A test/retest study of adaptive persistence



### Overview

This repository contains data, task code, and analysis code for one of our ongoing research projects. The primary research goal is identifying latent sources of individual differences in adaptive persistence using computational modeling methods. 

This project is supported by the Clara Mayo Memorial Fellowship administrated by the Psychological & Brain Sciences Department at Boston University (Recipient: Yixin Chen) and NIH Grant R21-MH124095 (PIs: Dr. Dan Fulford and Dr. Joseph T. McGuire). 


### Experiment Design
We conducted an online test/retest study with two sessions, around 3 weeks apart from each other. Data collection took place in two rounds. The first round started on 08/23/21. The second round started on 09/01/21. 

In total, 197 participants participated in the first session of our study. 182 of them who had satisfactorily completed the first session were invited to participate in the second session. 154 of them had satisfactorily completed both sessions.

In both sessions, participants were required to complete a 20-min behavioral task. In the first session, they were also required to complete three questionnaires:

- Monetary Choice Questionnaire
- UPPS-P
- BIS-II (Barratt impulsiveness scale) 

Besides, a subset of participants (those recruited in the second round) also completed the PANAS questionnaires (Positive and Negative Affect Schedule) in each session. 


### Data
Original files for both task data and self-report data are under ./analysis-code/data. 
- The task data file of each individual is named in the following format: task-SubjID-sessID.csv (e.g., task_s0001_sess1.csv).
- For each session, there is a header data file with basic task information for all participants (e.g., counterbalance, block durations, whether participants quit midway or not). 
- Self-report data for both sessions are saved as selfreport_sess1.csv and selfreport_sess2.csv.

### Intermediate Analysis Outputs
So far we've conducted some basic analyses and saved the outputs under ./analysis-code/analysis_log.  
- stats_sess1.csv and stats_sess2.csv contain some useful behavioral measures of task performance. 
- MCQ.csv contains discounting factor estimates based on the Monetary Choice Questionnaire. 
