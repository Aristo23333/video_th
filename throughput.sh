#num_prune = 5*8,10*8,11*8,15*8,20*8
python videomamba.py --model videomamba_small --num_prune 40 --output_path ./throughput/videomamba_small_40
python videomamba.py --model videomamba_small --num_prune 80 --output_path ./throughput/videomamba_small_80
python videomamba.py --model videomamba_small --num_prune 88 --output_path ./throughput/videomamba_small_88
python videomamba.py --model videomamba_small --num_prune 120 --output_path ./throughput/videomamba_small_120
python videomamba.py --model videomamba_small --num_prune 160 --output_path ./throughput/videomamba_small_160


