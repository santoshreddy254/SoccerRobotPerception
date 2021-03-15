#!/bin/bash
#SBATCH --job-name=train_nimbronet2
#SBATCH --partition=gpu
#SBATCH --nodes=1              # number of nodes
#SBATCH --mem=32GB               # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=2    # number of cores
#SBATCH --time=72:00:00           # HH-MM-SS
#SBATCH --output /home/smuthi2s/perl5/SoccerRobotPerception/cluster_logs/job_.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error /home/smuthi2s/perl5/SoccerRobotPerception/cluster_logs/job_.%N.%j.err  # filename for STDERR

# load cuda
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/rnd-tf

# locate to your root directory
cd /home/smuthi2s/perl5/SoccerRobotPerception

# run the script
python main.py --batch_size 8 --num_epochs 150 --dataset_path "/scratch/smuthi2s/data/" --save_path "/home/smuthi2s/perl5/SoccerRobotPerception/"
