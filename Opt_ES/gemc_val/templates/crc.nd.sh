{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#$ -N {{ id }}
#$ -pe smp {{ np_global }}
#$ -r n
#$ -m ae
#$ -q long
#$ -M mcarlozo@nd.edu
#$ -l h_rt=264:00:00

# module load gcc/9.1.0
# source /afs/crc.nd.edu/group/maginn/group_members/Ryan_DeFever/software/gromacs-2020/gromacs-dp/bin/GMXRC
module load gromacs
export PATH=/afs/crc.nd.edu/user/m/mcarlozo/.conda/envs/hfcs-fffit/bin:$PATH

{% block tasks %}
{% endblock %}
{% endblock %}

