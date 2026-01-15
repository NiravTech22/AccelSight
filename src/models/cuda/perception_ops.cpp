#include <torch/extension.h>
#include <vector>

// Forward declarations of launcher functions defined in .cu file
void launch_fast_nms(const float *objectness, float *output_mask,
                     int batch_size, int height, int width, float threshold);

void launch_coordinate_transform(const float *bboxes, const float *depth_map,
                                 float *world_coords, int batch_size,
                                 int height, int width, float focal_x,
                                 float focal_y, float center_x, float center_y,
                                 float grid_scale_x, float grid_scale_y);

// --- PyTorch Wrappers ---

torch::Tensor fast_nms(torch::Tensor objectness, float threshold) {
  TORCH_CHECK(objectness.is_cuda(), "objectness must be a CUDA tensor");

  auto obj_contig = objectness.contiguous();
  auto batch_size = obj_contig.size(0);
  auto height = obj_contig.size(2);
  auto width = obj_contig.size(3);
  auto output_mask = torch::zeros_like(obj_contig);

  launch_fast_nms(obj_contig.data_ptr<float>(), output_mask.data_ptr<float>(),
                  batch_size, height, width, threshold);

  return output_mask;
}

torch::Tensor coordinate_transform(torch::Tensor bboxes,
                                   torch::Tensor depth_map, float focal_x,
                                   float focal_y, float center_x,
                                   float center_y, float grid_scale_x,
                                   float grid_scale_y) {
  TORCH_CHECK(bboxes.is_cuda(), "bboxes must be a CUDA tensor");
  TORCH_CHECK(depth_map.is_cuda(), "depth_map must be a CUDA tensor");

  auto bboxes_contig = bboxes.contiguous();
  auto depth_contig = depth_map.contiguous();

  auto batch_size = bboxes_contig.size(0);
  auto height = bboxes_contig.size(2);
  auto width = bboxes_contig.size(3);
  auto world_coords =
      torch::zeros({batch_size, 3, height, width}, bboxes_contig.options());

  launch_coordinate_transform(
      bboxes_contig.data_ptr<float>(), depth_contig.data_ptr<float>(),
      world_coords.data_ptr<float>(), batch_size, height, width, focal_x,
      focal_y, center_x, center_y, grid_scale_x, grid_scale_y);

  return world_coords;
}

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_nms", &fast_nms, "Fast CUDA NMS (Heatmap-based)");
  m.def("coordinate_transform", &coordinate_transform,
        "CUDA-accelerated coordinate transformation");
}
