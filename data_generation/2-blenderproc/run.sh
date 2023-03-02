count=25

for i in $(seq $count); do
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_None/ 4 0
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_None/ 12 0
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_None/ 25 0
done

for i in $(seq $count); do
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_1/ 4 1
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_1/ 12 1
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_1/ 25 1
done

for i in $(seq $count); do
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_2/ 4 2
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_2/ 12 2
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_2/ 25 2
done

for i in $(seq $count); do
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_3/ 4 3
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_3/ 12 3
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_3/ 25 3
done

for i in $(seq $count); do
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_4/ 4 4
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_4/ 12 4
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_4/ 25 4
done

for i in $(seq $count); do
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_5/ 4 5
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_5/ 12 5
     blenderproc run examples/datasets/bop_object_on_surface_sampling/generate_dataset.py ./input tless resources/ output/texture_5/ 25 5
done