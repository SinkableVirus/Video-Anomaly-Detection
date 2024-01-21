import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from Seq2Seq import Seq2Seq
import cv2
from torch.utils.data import DataLoader
from PIL import Image
import os
from video_dataset2 import VideoDatasetWithFlows
import time
from sklearn.metrics import roc_auc_score
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_videos = []

test_video_paths = ["C:/Users/srini/OneDrive/Desktop/internship/avenue_vid/Avenue Dataset/testing_videos/{:02d}.avi".format(i) for i in range(1, 22)]
# test_video_paths = ["C:/Users/srini/OneDrive/Desktop/internship/avenue_vid/Avenue Dataset/testing_videos/01.avi"]

model = Seq2Seq(num_channels=1, num_kernels=64, kernel_size=(3, 3), padding=(1, 1), activation="relu", frame_size=(64, 64), num_layers=3).to(device)

optim = Adam(model.parameters(), lr=0)

criterion = nn.MSELoss()

checkpoint = torch.load("checkpoint_new_100.pth")
# model.load_state_dict(checkpoint)
model.load_state_dict(checkpoint["model_state"])

model.eval()

threshold = 32

root = 'C:/Users/srini/OneDrive/Desktop/internship/Attribute_based_VAD/Accurate-Interpretable-VAD/data/'
test_dataset = VideoDatasetWithFlows(dataset_name = 'avenue', root = root, train = False, normalize = False)


while threshold <= 32:
    since = time.time()
    result = []
    for video_path in test_video_paths:
        arr = []
        cap = cv2.VideoCapture(video_path)
        count = 0

        # print(video_path)
        
        while True:
            count = count + 1
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (64, 64))
            arr.append(resized_frame)

            if count > 10:
                target = arr[count - 1]
                target = torch.tensor(target)
                target = target.view(1, 1, 64, 64)
                target = target.to(device)

                inp = arr[count - 11 : count - 1]
                inp = torch.tensor(np.array(inp))
                inp = inp.view(1, 1, 10, 64, 64)
                inp = inp / 255.0
                inp = inp.to(device)
                # print(inp.shape)
                output = model(inp)

                output = output * 255

                loss = criterion(output, target)
                if count==11:
                    for i in range(11):
                        result.append(loss.detach().cpu().numpy())    
                else: 
                    result.append(loss.detach().cpu().numpy())
                # if loss < threshold:
                #     result.append(0)
                # else:
                #     result.append(1)

                # print("count: {}  Loss: {:.5f}".format(count, loss.item()))

            # elif count==11:
                # result.append(0)


        # print(f"Time taken: {time.time() - since:.4f} result len={len(result)}")
        cap.release()
        test_videos.append(np.array(arr))
    
    # result = torch.tensor(result)
    # print('threshold ', threshold , 'Micro AUC: ', roc_auc_score(test_dataset.all_gt, result) * 100)
    threshold = threshold + 1
    # print(f'result len={len(result)},gt len={test_dataset.num_of_frames},time={time.time()-since}')

    result=np.array(result)
    print(f'len of result={len(result)} and num of frames={test_dataset.num_of_frames}')
    for i in range(15):
        print(result[i])
    print(result)
    np.save('conv_lstm.npy',result)



# test_videos = np.array(test_videos)

# # np.random.shuffle(test_videos)

# def collate(batch):
#     max_length = len(batch[0])

#     padded_batch = torch.tensor(batch).unsqueeze(1)
#     padded_batch = padded_batch / 255.0
#     padded_batch = padded_batch.to(device)

#     return padded_batch[:, :, :-1], padded_batch[:, :, -1:]

# # Create a list to store the outputs for each frame
# outputs = []

# test_loader = DataLoader(test_videos, shuffle=True, batch_size=1, collate_fn=collate)



# for video_idx, batch in enumerate(test_loader):
#     image, target = batch

#     frame_number = 11

#     with torch.no_grad():
#         print(image.shape)
#         output = model(image)
#         loss = criterion(output, target)
#         outputs.append(output.cpu().numpy())

#         print("Video:", video_idx + 1, "Frame:", frame_number, "Loss:", loss.item())

#     frame_number += 1

# path = r"C:\Users\utkar\OneDrive\Desktop\SEM 4\CDSAML-CCBD\Internship Opportunity\anomaly\data\avenue\testing\frames" + "\\"
# vids = os.listdir(path)
# for i in vids:
#     image_path = path + i + "\\"
#     frames = os.listdir(image_path)
#     arr = []
#     count = 0
#     for j in frames:
#         count = count + 1
#         frame_path = image_path + j
#         print(frame_path)
#         frame = Image.open(frame_path)
#         frame = np.array(frame)
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         resized_frame = cv2.resize(gray_frame, (64, 64))
#         arr.append(resized_frame)
#         if count > 10:
#             print(count - 1, count - 11)
#             target = arr[count - 1]
#             target = torch.tensor(target)
#             target = target.view(1, 1, 64, 64)
#             target = target.to(device)

#             inp = arr[count - 11 : count - 1]
#             inp = torch.tensor(inp)
#             inp = inp.view(1, 1, 10, 64, 64)
#             inp = inp / 255.0
#             inp = inp.to(device)
#             # print(inp.shape)
#             output = model(inp)

#             output = output * 255

#             loss = criterion(output, target)

#             print(loss.item())





# Convert the outputs list to a numpy array
# outputs = np.concatenate(outputs)

# Process the outputs as needed
# ...

# Iterate over the frames and display the results
# for i in range(len(outputs)):
#     output_frame = outputs[i]
#     # Process the output_frame as needed
#     # ...
#     output_img = Image.fromarray(output_frame[0][0], "L")
#     output_img.show()
