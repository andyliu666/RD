for j in range(2):
    for i, (imgs, masks) in enumerate(dataloader_val):
        if j == 1:
            imgs, masks = imgs.cuda(), masks.cuda()
            print(imgs.shape)
            start_time = time.time()
            prediction = model(imgs)
            loss = loss_criterion(prediction, masks)
            val_loss += loss.item()
            print(" Val. Loss: {:.6f}".format(val_loss / (i + 1)))

            preds = prediction.squeeze().cpu().data.numpy()  ###从gpu取
            print(preds.shape)
            imgs = imgs.cpu().data.numpy()

            gts = masks.squeeze().cpu().data.numpy().astype('uint8')

            preds = preds.round().astype('uint8')

            imgs = imgs.transpose(0, 2, 3, 1)
            for i in range(preds.shape[0]):
                pred = img_to_rgb(preds[i])
                fig, ax = plt.subplots(1)

                mask = pred

        # np.nonzero会返回形如(array,array)的东西
        # 第一个是行的index，第二个是列的index
        # 例如 np.nonzero(np.array([[0,1],[2,0]])
        # 会返回 ( array([0,1]), array([1,0]) )
                coor = np.nonzero(mask)
                xmin = coor[0][0]
                xmax = coor[0][-1]
                coor[1].sort()  # 直接改变原数组，没有返回值
                ymin = coor[1][0]
                ymax = coor[1][-1]

                bottomleft = (ymin, xmin)

                width = ymax - ymin
                height = xmax - xmin
                total += time.time() - start_time
        else:
            imgs, masks = imgs.cuda(), masks.cuda()
            print(imgs.shape)
            start_time = time.time()
            prediction = model(imgs)
            loss = loss_criterion(prediction, masks)
            val_loss += loss.item()
            print(" Val. Loss: {:.6f}".format(val_loss / (i + 1)))

            preds = prediction.squeeze().cpu().data.numpy()  ###从gpu取
            print(preds.shape)
            imgs = imgs.cpu().data.numpy()

            gts = masks.squeeze().cpu().data.numpy().astype('uint8')

            preds = preds.round().astype('uint8')

            imgs = imgs.transpose(0, 2, 3, 1)
            for i in range(preds.shape[0]):
                pred = img_to_rgb(preds[i])
                fig, ax = plt.subplots(1)

                mask = pred

                # np.nonzero会返回形如(array,array)的东西
                # 第一个是行的index，第二个是列的index
                # 例如 np.nonzero(np.array([[0,1],[2,0]])
                # 会返回 ( array([0,1]), array([1,0]) )
                coor = np.nonzero(mask)
                xmin = coor[0][0]
                xmax = coor[0][-1]
                coor[1].sort()  # 直接改变原数组，没有返回值
                ymin = coor[1][0]
                ymax = coor[1][-1]

                bottomleft = (ymin, xmin)

                width = ymax - ymin
                height = xmax - xmin
            #print(xmin*4.6875,xmax*4.6875,ymin*7.5,ymax*7.5)
print(total)
