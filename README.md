# Intro
- I wish to start by saying that was a challenging and very informative test for me, since I haven't worked on neither medical related data nor real ML datasets, i managed to make a script with the bit of knowledge i still had from my masters courses, and also some help from chatGPT.
- First step of the the script is to perfom data exploration, and generate some simple graphs and plots in .pdf format. You will find them all in the results dir.
- The script in python uses the scikit learn library to train and evaluate 2 simple models : Logistic regression(Elastic-Net Multinomial Logistic Regression) and decision trees(HistGradientBoostingClassifier (HGB)).

# Approach and Rationale
- In the 2nd part of the assessment i was limited by knowledge and experience, so the coices of models and libraries were made out of convenience since they are all i know about.
- The dataset is also limited so going simple is the best practice.
  - So i chose 1 linear model beacause it's the most simple + 1 non linear model because we have 3 classes of classifaction and linear models are not so good outside binary classes.
  - This is the first times i use these specific models (Elastic-Net Multinomial Logistic Regression and HistGradientBoostingClassifier); they seemed to fit what i wanted to do and simple enough for me to use and understand a bit.
- After data exploration there seems to be no data missing, no class of the 3 is represented a lot more than the others, so i proceeded.

# Results & discussion
- To compare the 2 models i looked at the :
      - Balanced accuracy : is the average recall across classes .
      - Macro F1 :averages F1 scores of all classes equally.
      - then class by class metrics
- All of these metrics are summarized  in a pdf : model_full_reports.pdf

- By the looks of it the linear model performed better than the decison trees, it could be because the dataset is to simple and the decison tree model increases complexity and overfits.
  
# Next steps
- What i would try is to compare the datasets 2 by 2, since we see in exploratory data that AD+MCI >> CN. Then maybe comparing CN+MCI to AD would give better results.
