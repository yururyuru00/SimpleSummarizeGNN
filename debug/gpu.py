import subprocess
import csv

default_properies = (
  "timestamp",
  "gpu_name",
  #"gpu_uuid",
  "index",
  "memory.total",
  "memory.used",
  "memory.free",
  "utilization.gpu",
  "utilization.memory",
)
 
def dump_gpu_properties(
  cmd_path="nvidia-smi",
  target_properties=default_properies,
  noheader=True,
  nounits=True,
  save_name="gpu.csv"
  ):
  """
  CUDA GPUのプロパティ情報取得
 
  Parameters
  ----------
  cmd_path : str
    コマンドラインから"nvidia-smi"を実行する際のパス
  target_properties : obj
    取得するプロパティ情報
    プロパティ情報の詳細は"nvidia-smi --help-query-gpu"で取得可能
  noheader : bool
    skip the first line with column headers
  nounits : bool
    don't print units for numerical values
 
  Returns
  -------
  gpu_properties : list
    gpuごとのproperty情報
  """
    
  # formatオプション定義
  format_option = "--format=csv"
  if noheader:
      format_option += ",noheader"
  if nounits:
      format_option += ",nounits"
 
  # コマンド生成
  cmd = '%s --query-gpu=%s %s' % (cmd_path, ','.join(target_properties), format_option)
 
  # サブプロセスでコマンド実行
  cmd_res = subprocess.check_output(cmd, shell=True)
    
  # コマンド実行結果をオブジェクトに変換
  gpu_lines = cmd_res.decode().split('\n')
  # リストの最後の要素に空行が入るため除去
  gpu_lines = [ line.strip() for line in gpu_lines if line.strip() != '' ]
 
  # ", "ごとにプロパティ情報が入っているのでdictにして格納
  gpu_properties = [ { k: v for k, v in zip(target_properties, line.split(', ')) } for line in gpu_lines ]

  with open(save_name,"w") as outfile:
    writerfile = csv.writer(outfile)
    writerfile.writerow(gpu_properties.keys())
    writerfile.writerows(zip(*gpu_properties.values()))
