#include <chrono>
#include <string>
#include <regex>
#include <algorithm>
#include "boost/range.hpp"
#include "boost/filesystem.hpp"
#include "is/msgs/image.pb.h"
#include "google/protobuf/wrappers.pb.h"
#include "is/wire/core/logger.hpp"
#include "skeletons_grouper.hpp"
#include "stream_pb.hpp"
#include "vision.hpp"
#include "options.pb.h"

namespace fs = boost::filesystem;
using namespace std::chrono;
using namespace google::protobuf;

auto const base_folder = fs::path("/home/felippe/skeletons-data");

template <class Container, class T>
bool find_id(Container c, T value) {
  return std::find(c.begin(), c.end(), value) != c.end();
}

std::vector<int64_t> get_cameras(fs::path const& dataset_dir, std::string const& model) {
  std::vector<int64_t> cameras;
  auto const id_regex = std::regex(fmt::format("{}_2d_detector_(\\d+)?", model));
  for (auto& entry : boost::make_iterator_range(fs::directory_iterator(dataset_dir.string()), {})) {
    auto filename = entry.path().filename().string();
    std::smatch matches;
    if (std::regex_match(filename, matches, id_regex)) { cameras.push_back(std::stoi(matches[1])); }
  }
  std::sort(cameras.begin(), cameras.end());
  return cameras;
}

int main(int argc, char** argv) {
  if (argc > 2) is::critical("Enter an options file: ./group_datasest <OPTIONS_FILE[options.json]>");

  auto filename = argc == 2 ? argv[1] : "options.json";
  is::DatasetSkeletonsGrouperOptions options;
  auto status = is::load(filename, &options);
  if (status.code() != is::wire::StatusCode::OK) is::critical("{}", status);
  is::info("Options: \n{}", options);

  auto sufix = is::SourceType_Name(options.source_type());
  std::transform(sufix.begin(), sufix.end(), sufix.begin(), ::tolower);
  auto const dataset_folder = fs::path(options.basedir()) / fs::path(options.dataset());
  auto model = is::SkeletonModel_Name(options.model());
  std::transform(model.begin(), model.end(), model.begin(), ::tolower);
  auto cameras = get_cameras(dataset_folder, model);
  auto const calibrs_folder = dataset_folder / fs::path("calibrations");
  auto calibrations = load_calibs(calibrs_folder.string());
  for (auto& kv : calibrations) {
    is::info("[Calibration] {}", kv.first);
    kv.second.PrintDebugString();
  }

  SkeletonsGrouper grouper(calibrations, options.referencial(), options.min_error());

  auto const detections_file = dataset_folder / fs::path(fmt::format("{}_2d_{}", model, sufix));
  ProtobufReader reader(detections_file.string());
  auto const output_file = dataset_folder / fs::path(fmt::format("{}_3d_{}_grouped", model, sufix));
  ProtobufWriter writer(output_file.string());
  auto const time_output_file = dataset_folder / fs::path(fmt::format("{}_duration_{}_grouped", model, sufix));
  ProtobufWriter time_writer(time_output_file.string());

  is::info("[IN][Detections] {}", detections_file.string());
  is::info("[OUT][Detections] {}", output_file.string());
  is::info("[OUT][Durations] {}", time_output_file.string());

  for (;;) {
    auto sequence_id = reader.next<Int64Value>();
    if (!sequence_id) break;

    std::unordered_map<int64_t, is::vision::ObjectAnnotations> sks_2d;
    for (auto& camera : cameras) {
      auto objs = reader.next<is::vision::ObjectAnnotations>();
      if (!objs) is::critical("Failed on read \'ObjectAnnotations\'");
      sks_2d[camera] = *objs;
    }

    auto t0 = system_clock::now();
    auto sks_3d = grouper.group(sks_2d);
    auto tf = system_clock::now();
    auto dt_ms = duration_cast<microseconds>(tf - t0).count() / 1000.0;

    is::info("[{:>5d}] took {:.2f}ms and detect {} skeletons", sequence_id->value(), dt_ms, sks_3d.objects().size());

    writer.insert(*sequence_id);
    writer.insert(sks_3d);
    time_writer.insert(*sequence_id);
    time_writer.insert(is::to_duration(tf - t0));
  }

  return 0;
}