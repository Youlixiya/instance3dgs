#fern
CUDA_VISIBLE_DEVICES=0 python train_gs.py --source_path data/nerf_llff_data/fern --images images_4 --model_path output/llff/fern

CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/fern/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/fern/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/fern/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4 --global_contrastive

CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/fern/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/fern/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4 --gs_feature_dim 8
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/fern/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4 --gs_feature_dim 32
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/fern/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4 --no_cache
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/fern/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/fern/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4 --global_contrastive --instance_feature_dim 8

CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/fern/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/fern/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/fern/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4 --global_contrastive --depth_decode
python viewer.py --gs_source output/llff/fern/point_cloud/iteration_10000/point_cloud.ply  --colmap_dir data/nerf_llff_data/fern --feature_gs_source output/llff/fern/point_cloud/iteration_10000/16_feature_gs_10000.pt --images images_4 --lisa
python test_llff.py --cfg_path scripts/16_llff_test_config.json  --scene fern --images images_4

#flower
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/nerf_llff_data/flower --images images_4 --model_path output/llff/flower

CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/flower/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/flower --images images_4
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/flower/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/flower --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/flower/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/flower --images images_4 --global_contrastive

CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/flower/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/flower --images images_4
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/flower/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/flower --images images_4 --gs_feature_dim 8
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/flower/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/flower --images images_4 --gs_feature_dim 32
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/flower/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/flower --images images_4 --no_cache
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/flower/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/flower --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/flower/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/flower --images images_4 --global_contrastive --instance_feature_dim 8

CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/flower/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/flower --images images_4
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/flower/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/flower --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/flower/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/flower --images images_4 --global_contrastive --depth_decode
python viewer.py --gs_source output/llff/flower/point_cloud/iteration_5000/point_cloud.ply  --colmap_dir data/nerf_llff_data/flower --feature_gs_source output/llff/flower/point_cloud/iteration_5000/batch_contrastive_feature_gs_10000.pt --gemini --images images_4
python test_nvos.py --cfg_path scripts/batch_contrastive_llff_test_config.json  --scene flower --images images_4
python test_nvos.py --cfg_path scripts/queue_contrastive_llff_test_config.json  --scene flower --images images_4
python test_nvos.py --cfg_path scripts/global_contrastive_llff_test_config.json  --scene flower --images images_4

#fortress
CUDA_VISIBLE_DEVICES=2 python train_gs.py --source_path data/nerf_llff_data/fortress --images images_4 --model_path output/llff/fortress

CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/fortress/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/fortress/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/fortress/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4 --global_contrastive

CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/fortress/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/fortress/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4 --gs_feature_dim 8
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/fortress/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4 --gs_feature_dim 32
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/fortress/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4 --no_cache
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/fortress/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/fortress/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4 --global_contrastive --instance_feature_dim 8


CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/fortress/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/fortress/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/fortress/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4 --global_contrastive --depth_decode
python viewer.py --gs_source output/llff/fortress/point_cloud/iteration_5000/point_cloud.ply  --colmap_dir data/nerf_llff_data/fortress --feature_gs_source output/llff/fortress/point_cloud/iteration_5000/batch_contrastive_feature_gs_10000.pt --gemini --images images_4
python test_nvos.py --cfg_path scripts/batch_contrastive_llff_test_config.json  --scene fortress --images images_4
python test_nvos.py --cfg_path scripts/queue_contrastive_llff_test_config.json  --scene fortress --images images_4
python test_nvos.py --cfg_path scripts/global_contrastive_llff_test_config.json  --scene fortress --images images_4

#horns
CUDA_VISIBLE_DEVICES=3 python train_gs.py --source_path data/nerf_llff_data/horns --images images_4 --model_path output/llff/horns

CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/horns/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/horns/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/horns/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4 --global_contrastive

CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/horns/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/horns/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4 --gs_feature_dim 8
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/horns/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4 --gs_feature_dim 32
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/horns/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4 --no_cache
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/horns/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/horns/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4 --global_contrastive --instance_feature_dim 8

CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/horns/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/horns/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/horns/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4 --global_contrastive --depth_decode
python viewer.py --gs_source output/llff/horns/point_cloud/iteration_10000/point_cloud.ply  --colmap_dir data/nerf_llff_data/horns --feature_gs_source output/llff/horns/point_cloud/iteration_10000/8_feature_gs_10000.pt --images images_4
python test_nvos.py --cfg_path scripts/batch_contrastive_llff_test_config.json  --scene horns_center --images images_4
python test_nvos.py --cfg_path scripts/queue_contrastive_llff_test_config.json  --scene horns_center --images images_4
python test_nvos.py --cfg_path scripts/global_contrastive_llff_test_config.json  --scene horns_center --images images_4

python test_nvos.py --cfg_path scripts/batch_contrastive_llff_test_config.json  --scene horns_left --images images_4
python test_nvos.py --cfg_path scripts/queue_contrastive_llff_test_config.json  --scene horns_left --images images_4
python test_nvos.py --cfg_path scripts/global_contrastive_llff_test_config.json  --scene horns_left --images images_4

#leaves
CUDA_VISIBLE_DEVICES=4 python train_gs.py --source_path data/nerf_llff_data/leaves --images images_4 --model_path output/llff/leaves

CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/llff/leaves/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4
CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/llff/leaves/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/llff/leaves/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4 --global_contrastive

CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/llff/leaves/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4
CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/llff/leaves/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4 --gs_feature_dim 8
CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/llff/leaves/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4 --gs_feature_dim 32
CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/llff/leaves/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4 --no_cache
CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/llff/leaves/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/llff/leaves/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4 --global_contrastive --instance_feature_dim 8

CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/llff/leaves/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4
CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/llff/leaves/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/llff/leaves/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4 --global_contrastive --depth_decode
python viewer.py --gs_source output/llff/leaves/point_cloud/iteration_5000/point_cloud.ply  --colmap_dir data/nerf_llff_data/leaves --feature_gs_source output/llff/leaves/point_cloud/iteration_5000/batch_contrastive_feature_gs_10000.pt --gemini --images images_4
python test_nvos.py --cfg_path scripts/batch_contrastive_llff_test_config.json  --scene leaves --images images_4
python test_nvos.py --cfg_path scripts/queue_contrastive_llff_test_config.json  --scene leaves --images images_4
python test_nvos.py --cfg_path scripts/global_contrastive_llff_test_config.json  --scene leaves --images images_4
#orchids
CUDA_VISIBLE_DEVICES=5 python train_gs.py --source_path data/nerf_llff_data/orchids --images images_4 --model_path output/llff/orchids 

CUDA_VISIBLE_DEVICES=5 python train.py --gs_source output/llff/orchids/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4 
CUDA_VISIBLE_DEVICES=5 python train.py --gs_source output/llff/orchids/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=5 python train.py --gs_source output/llff/orchids/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4 --global_contrastive 

CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/orchids/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/orchids/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4 --gs_feature_dim 8
CUDA_VISIBLE_DEVICES=5 python train.py --gs_source output/llff/orchids/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4 --gs_feature_dim 32
CUDA_VISIBLE_DEVICES=5 python train.py --gs_source output/llff/orchids/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4 --no_cache
CUDA_VISIBLE_DEVICES=5 python train.py --gs_source output/llff/orchids/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=5 python train.py --gs_source output/llff/orchids/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4 --global_contrastive --instance_feature_dim 8

CUDA_VISIBLE_DEVICES=5 python train.py --gs_source output/llff/orchids/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4 
CUDA_VISIBLE_DEVICES=5 python train.py --gs_source output/llff/orchids/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=5 python train.py --gs_source output/llff/orchids/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4 --global_contrastive --depth_decode
python viewer.py --gs_source output/llff/orchids/point_cloud/iteration_5000/point_cloud.ply  --colmap_dir data/nerf_llff_data/orchids --feature_gs_source output/llff/orchids/point_cloud/iteration_5000/batch_contrastive_feature_gs_10000.pt --images images_4
python test_nvos.py --cfg_path scripts/batch_contrastive_llff_test_config.json  --scene orchids --images images_4
python test_nvos.py --cfg_path scripts/queue_contrastive_llff_test_config.json  --scene orchids --images images_4
python test_nvos.py --cfg_path scripts/global_contrastive_llff_test_config.json  --scene orchids --images images_4

#room
CUDA_VISIBLE_DEVICES=0 python train_gs.py --source_path data/nerf_llff_data/room --images images_4 --model_path output/llff/room

CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/room/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/room/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/room/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4 --global_contrastive

CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/room/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/room/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4 --gs_feature_dim 8
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/room/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4 --gs_feature_dim 32
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/room/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4 --no_cache
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/room/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/room/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4 --global_contrastive --instance_feature_dim 8

CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/room/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/room/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/llff/room/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4 --global_contrastive --depth_decode
python viewer.py --gs_source output/llff/room/point_cloud/iteration_5000/point_cloud.ply  --colmap_dir data/nerf_llff_data/room --feature_gs_source output/llff/room/point_cloud/iteration_5000/batch_contrastive_feature_gs_10000.pt --gemini --images images_4
python test.py --gs_source output/llff/room/point_cloud/iteration_5000/point_cloud.ply  --colmap_dir data/nerf_llff_data/room --feature_gs_source output/llff/room/point_cloud/iteration_5000/batch_contrastive_feature_gs_10000.pt --images images_4

#trex
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/nerf_llff_data/trex --images images_4 --model_path output/llff/trex

CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/trex/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/trex --images images_4
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/trex/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/trex --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/trex/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/trex --images images_4 --global_contrastive

CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/trex/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/trex --images images_4
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/trex/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/trex --images images_4 --gs_feature_dim 8
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/trex/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/trex --images images_4 --gs_feature_dim 32
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/trex/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/trex --images images_4 --no_cache
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/trex/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/trex --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/trex/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/trex --images images_4 --global_contrastive --instance_feature_dim 8

CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/trex/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/trex --images images_4
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/trex/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/trex --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/trex/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/trex --images images_4 --global_contrastive --depth_decode
python viewer.py --gs_source output/llff/trex/point_cloud/iteration_5000/point_cloud.ply  --colmap_dir data/nerf_llff_data/trex --feature_gs_source output/llff/trex/point_cloud/iteration_5000/batch_contrastive_feature_gs_10000.pt --gemini --images images_4
python test.py --cfg_path scripts/batch_contrastive_llff_test_config.json  --scene trex --images images_4
python test.py --cfg_path scripts/queue_contrastive_llff_test_config.json  --scene trex --images images_4
python test.py --cfg_path scripts/global_contrastive_llff_test_config.json  --scene trex --images images_4

python metrics_nvos.py --cfg_path scripts/batch_contrastive_llff_test_config.json --gt_path data/masks --save_tag batch_contrastive
python metrics_nvos.py --cfg_path scripts/queue_contrastive_llff_test_config.json --gt_path data/masks --save_tag queue_contrastive
python metrics_nvos.py --cfg_path scripts/global_contrastive_llff_test_config.json --gt_path data/masks --save_tag global_contrastive

python render.py --cfg_path scripts/render_config.json  --scene horns