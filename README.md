## Requirements

```
numpy >= 1.20.1
pandas >= 1.2.4
scipy >= 1.6.2
python >= 3.8
```


## Introduction

Below is the directory structure of this project.

```plaintext
DP-BGS/
├── datasets/                         # Datasets used in the project
│   ├── book.txt     
│   ├── movie.txt   
│   ├── pantry.txt       
│   ├── yahoo.txt    
├── results/                         # The folder used for saving results
│   ├── book/  
│   ├── movie/
│   ├── pantry/  
│   ├── yahoo/
├── eval1.py                         # Used for evaluation
├── generate1_ab.py                  # Synthesizing the bipartite graph using BKGS or DBGS
├── generate1.py                     # Synthesizing the bipartite graph using JBGS
├── main_BKGS.py                     # Main function for BKGS
├── main_DBGS.py                     # Main function for DBGS
├── main_JBGS.py                     # Main function for JBGS        
├── README.md                        # Main project documentation                                           
├── utils.py                         # Helper classes and functions supporting various operations                  
```


## Running

We provide an example of synthesizing the bipartite graph using JBGS.
```
python main_JBGS.py --dataset pantry --epsilon 0.5 --seed_n 1
```


Similarly, we provide an example of synthesizing the bipartite graph using BKGS.
```
python main_BKGS.py --dataset pantry --epsilon 0.5 --seed_n 1
```

The DBGS is similar to BKGS.
```
python main_DBGS.py --dataset pantry --epsilon 0.5 --seed_n 1
```





