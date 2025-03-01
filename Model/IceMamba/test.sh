# IceMamba-1
python IceMamba_test.py --num_forecast 1 --ckpt_path ./ckpt/best_sm_icemamba-1.ckpt --time_dict_path ../../train_json/time_dict1.json --preprocess_dict_path ../../train_json/preprocess_dict1.json --result_save_name result_IceMamba-1
# IceMamba-4
python IceMamba_test.py --num_forecast 4 --ckpt_path ./ckpt/best_sm_icemamba-4.ckpt --time_dict_path ../../train_json/time_dict2.json --preprocess_dict_path ../../train_json/preprocess_dict2.json --result_save_name result_IceMamba-4
# IceMamba-6
python IceMamba_test.py --num_forecast 6 --ckpt_path ./ckpt/best_sm_icemamba-6.ckpt --time_dict_path ../../train_json/time_dict8.json --preprocess_dict_path ../../train_json/preprocess_dict8.json --result_save_name result_IceMamba-6
# IceMamba-1-ERA5
python IceMamba_test.py --num_forecast 1 --ckpt_path ./ckpt/best_sm_icemamba-1-ERA5.ckpt --time_dict_path ../../train_json/time_dict0.json --preprocess_dict_path ../../train_json/preprocess_dict0.json --result_save_name result_IceMamba-1-ERA5
# IceMamba-4-ERA4
python IceMamba_test.py --num_forecast 4 --ckpt_path ./ckpt/best_sm_icemamba-4-ERA5.ckpt --time_dict_path ../../train_json/time_dict0.json --preprocess_dict_path ../../train_json/preprocess_dict0.json --result_save_name result_IceMamba-4-ERA5
# IceMamba-6-ERA4
python IceMamba_test.py --num_forecast 6 --ckpt_path ./ckpt/best_sm_icemamba-6-ERA5.ckpt --time_dict_path ../../train_json/time_dict0.json --preprocess_dict_path ../../train_json/preprocess_dict0.json --result_save_name result_IceMamba-6-ERA5
# IceMamba-6-ua10
python IceMamba_test.py --num_forecast 6 --ckpt_path ./ckpt/best_sm_icemamba-6-ua10.ckpt --time_dict_path ../../train_json/time_dict_ua10.json --preprocess_dict_path ../../train_json/preprocess_dict_ua10.json --result_save_name result_IceMamba-6-ua10
# IceMamba-6-VSSB
python IceMamba_6_VSSB_test.py --num_forecast 6 --ckpt_path ./ckpt/best_sm_icemamba-6-VSSB.ckpt --time_dict_path ../../train_json/time_dict8.json --preprocess_dict_path ../../train_json/preprocess_dict8.json --result_save_name result_IceMamba-6-VSSB
