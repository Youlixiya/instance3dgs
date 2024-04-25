python demo/demo_automatic.py --chunk_size 4 --img_path ../data/tandt/truck/images --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ./example/masks
python demo/demo_automatic.py --chunk_size 4 --img_path ../data/lego_real_night_radial/images --amp --temporal_setting semionline --size 480 --output ./example/masks
python demo/demo_with_text.py --chunk_size 4 --img_path ../data/lego_real_night_radial/images --amp --temporal_setting semionline --size 480 --output ./example/bucket --prompt bucket.
python demo/demo_automatic.py --chunk_size 4 --img_path ../data/nerf_real_360/pinecone/images_8 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ./example/masks_8
python demo/demo_automatic.py --chunk_size 4 --img_path ../data/fork/images_8 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ./example/masks_8
CUDA_VISIBLE_DEVICES=4 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/ovs3d/sofa/images_4 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/ovs3d/sofa/masks_4
CUDA_VISIBLE_DEVICES=4 python demo/demo_with_text.py --chunk_size 4 --img_path ../data/ovs3d/sofa/images_4 --amp --temporal_setting semionline --size 480 --output ./example/gundam --prompt gundam. --DINO_THRESHOLD 0.72
CUDA_VISIBLE_DEVICES=4 python demo/demo_with_text.py --chunk_size 4 --img_path ../data/ovs3d/sofa/images_4 --amp --temporal_setting semionline --size 480 --output ./example/gray_sofa --prompt "gray sofa". --DINO_THRESHOLD 0.72
CUDA_VISIBLE_DEVICES=5 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/ovs3d/bed/images_4 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/ovs3d/bed/masks_4
CUDA_VISIBLE_DEVICES=5 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/ovs3d/bench/images_4 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/ovs3d/bench/masks_4
CUDA_VISIBLE_DEVICES=5 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/ovs3d/lawn/images_4 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/ovs3d/lawn/masks_4
CUDA_VISIBLE_DEVICES=5 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/ovs3d/room/images_4 --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/ovs3d/room/masks_4
CUDA_VISIBLE_DEVICES=5 python demo/demo_with_text.py --chunk_size 4 --img_path ../data/ovs3d/room/images_4 --amp --temporal_setting semionline --size 480 --output ./example/background --prompt "background." --DINO_THRESHOLD 0.72


CUDA_VISIBLE_DEVICES=1 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/lerf_data/figurines/images --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/lerf_data/figurines/masks
CUDA_VISIBLE_DEVICES=1 python demo/demo_automatic.py --chunk_size 4 --img_path ../data/lerf_data/teatime/images --amp --temporal_setting semionline --size 480 --suppress_small_objects --output ../data/lerf_data/teatime/masks