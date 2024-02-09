import logging
import time

import joblib
import numpy as np
import pandas as pd
import yaml

MODEL_DIR = "models"


def test_model_correctness(data):
    """
    Thử nghiệm độ chính xác của mô hình.

    Args:
        data: Dữ liệu đầu vào.

    Returns:
        Hai giá trị:
            * True nếu mô hình dự đoán đúng là người không bị tiểu đường.
            * True nếu mô hình dự đoán đúng là người bị tiểu đường.
    """

    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    clf = joblib.load(f"{MODEL_DIR}/diabetes_model.pkl")
    x = np.array(data).reshape(-1, 8)
    x = scaler.transform(x)
    pred = clf.predict(x)[0]

    return pred == 0, pred == 1


def main():
    """
    Chương trình chính.
    """

    logging.basicConfig(level=logging.INFO)

    # Khởi tạo dữ liệu đầu vào
    # data = [
    #     -1.15332192,
    #     -0.05564105,
    #     0.12035144,
    #     -1.25882277,
    #     -1.08285125,
    #     -0.28446352,
    #     -0.49468374,
    #     -0.52559768,
    # ]

    data = [
            2.0,
            127.0,
            46.0,
            21.0,
            335.0,
            34.4,
            0.176,
            22.0
        ],

    # Thử nghiệm độ chính xác của mô hình
    start_time = time.time()
    is_correct_not_diabetes, is_correct_diabetes = test_model_correctness(data)
    end_time = time.time()

    # Hiển thị thông báo kết quả
    if is_correct_not_diabetes:
        logging.info("Mô hình dự đoán chính xác người không bị tiểu đường.")
    else:
        logging.info("Mô hình dự đoán không chính xác người không bị tiểu đường.")

    if is_correct_diabetes:
        logging.info("Mô hình dự đoán chính xác người bị tiểu đường.")
    else:
        logging.info("Mô hình dự đoán không chính xác người bị tiểu đường.")

    # Hiển thị thông số đánh giá
    logging.info("Độ chính xác người không bị tiểu đường: %.2f%%", 100 * is_correct_not_diabetes)
    logging.info("Độ chính xác người bị tiểu đường: %.2f%%", 100 * is_correct_diabetes)


if __name__ == "__main__":
    main()






# import logging
# import time

# import joblib
# import numpy as np
# import pandas as pd
# import yaml

# MODEL_DIR = "models"


# def test_model_correctness(data):
#     """
#     Thử nghiệm độ chính xác của mô hình.

#     Args:
#         data: Dữ liệu đầu vào.

#     Returns:
#         True nếu mô hình dự đoán chính xác, False nếu không.
#     """

#     scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
#     clf = joblib.load(f"{MODEL_DIR}/diabetes_model.pkl")
#     x = np.array(data).reshape(-1, 8)
#     x = scaler.transform(x)
#     pred = clf.predict(x)[0]
   
#     return pred == 0, pred


# def main():
#     """
#     Chương trình chính.
#     """

#     logging.basicConfig(level=logging.INFO)

#     # Khởi tạo dữ liệu đầu vào
#     data = [
#         -1.15332192,
#         -0.05564105,
#         0.12035144,
#         -1.25882277,
#         -1.08285125,
#         -0.28446352,
#         -0.49468374,
#         -0.52559768,
#     ]

#     # Thử nghiệm độ chính xác của mô hình
#     start_time = time.time()
#     is_correct, pred = test_model_correctness(data)
#     end_time = time.time()

#     # Hiển thị thông báo kết quả
#     if is_correct:
#         logging.info("Mô hình dự đoán chính xác.")
#         logging.info(f"Thời gian thực hiện: {end_time - start_time:.2f} giây.")
#         logging.info("Kết quả dự đoán: {}".format(pred))
#     else:
#         logging.info("Mô hình dự đoán không chính xác.")

#     # Hiển thị thông số đánh giá
#     logging.info("Độ chính xác: %.2f%%", 100 * is_correct)


# if __name__ == "__main__":
#     main()



# import logging
# import time

# import joblib
# import numpy as np
# import pandas as pd
# import yaml

# MODEL_DIR = "models"


# def test_model_correctness():
#     scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
#     clf = joblib.load(f"{MODEL_DIR}/diabetes_model.pkl")
#     data = [
#         -1.15332192,
#         -0.05564105,
#         0.12035144,
#         -1.25882277,
#         -1.08285125,
#         -0.28446352,
#         -0.49468374,
#         -0.52559768,
#     ]
#     x = np.array(data).reshape(-1, 8)
#     x = scaler.transform(x)
#     pred = clf.predict(x)[0]
#     assert pred == 0

# test_model_correctness()