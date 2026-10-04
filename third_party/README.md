# 可选 Open-Jev 集成

上游：https://github.com/Zefan-Cai/Open-Jev

本机检出的版本为 `3308a15ccd7eea1df7a37d6ddc39b023b801ba16`。
仓库只保存本项目对 `jev/model.py`、`jev/server.py`、`jev/serving.py`
的修改补丁，不重复上传上游代码和生成报告。

```powershell
git clone https://github.com/Zefan-Cai/Open-Jev.git third_party/Open-Jev
git -C third_party/Open-Jev checkout 3308a15ccd7eea1df7a37d6ddc39b023b801ba16
git -C third_party/Open-Jev apply ../openjev-local.patch
```

安装与模型配置见上游说明和 `docs/temporal_forecast_and_open_jev.md`。
