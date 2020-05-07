DAF3D(
  (backbone): BackBone3D(
    (layer0): Sequential(
      (0): Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
      (1): GroupNorm(32, 64, eps=1e-05, affine=True)
      (2): PReLU(num_parameters=1)
    )
    (layer1): Sequential(
      (0): MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1, dilation=1, ceil_mode=False)
      (1): Sequential(
        (0): ResNeXtBottleneck(
          (conv1): Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
          (gn1): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=32, bias=False)
          (gn2): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv3): Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
          (gn3): GroupNorm(32, 256, eps=1e-05, affine=True)
          (relu): PReLU(num_parameters=1)
          (downsample): Sequential(
            (0): Conv3d(64, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (1): GroupNorm(32, 256, eps=1e-05, affine=True)
          )
        )
        (1): ResNeXtBottleneck(
          (conv1): Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
          (gn1): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=32, bias=False)
          (gn2): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv3): Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
          (gn3): GroupNorm(32, 256, eps=1e-05, affine=True)
          (relu): PReLU(num_parameters=1)
        )
        (2): ResNeXtBottleneck(
          (conv1): Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
          (gn1): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=32, bias=False)
          (gn2): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv3): Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
          (gn3): GroupNorm(32, 256, eps=1e-05, affine=True)
          (relu): PReLU(num_parameters=1)
        )
      )
    )
    (layer2): Sequential(
      (0): ResNeXtBottleneck(
        (conv1): Conv3d(256, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn1): GroupNorm(32, 256, eps=1e-05, affine=True)
        (conv2): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), groups=32, bias=False)
        (gn2): GroupNorm(32, 256, eps=1e-05, affine=True)
        (conv3): Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn3): GroupNorm(32, 512, eps=1e-05, affine=True)
        (relu): PReLU(num_parameters=1)
        (downsample): Sequential(
          (0): Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False)
          (1): GroupNorm(32, 512, eps=1e-05, affine=True)
        )
      )
      (1): ResNeXtBottleneck(
        (conv1): Conv3d(512, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn1): GroupNorm(32, 256, eps=1e-05, affine=True)
        (conv2): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=32, bias=False)
        (gn2): GroupNorm(32, 256, eps=1e-05, affine=True)
        (conv3): Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn3): GroupNorm(32, 512, eps=1e-05, affine=True)
        (relu): PReLU(num_parameters=1)
      )
      (2): ResNeXtBottleneck(
        (conv1): Conv3d(512, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn1): GroupNorm(32, 256, eps=1e-05, affine=True)
        (conv2): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=32, bias=False)
        (gn2): GroupNorm(32, 256, eps=1e-05, affine=True)
        (conv3): Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn3): GroupNorm(32, 512, eps=1e-05, affine=True)
        (relu): PReLU(num_parameters=1)
      )
      (3): ResNeXtBottleneck(
        (conv1): Conv3d(512, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn1): GroupNorm(32, 256, eps=1e-05, affine=True)
        (conv2): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=32, bias=False)
        (gn2): GroupNorm(32, 256, eps=1e-05, affine=True)
        (conv3): Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn3): GroupNorm(32, 512, eps=1e-05, affine=True)
        (relu): PReLU(num_parameters=1)
      )
    )
    (layer3): Sequential(
      (0): ResNeXtDilatedBottleneck(
        (conv1): Conv3d(512, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn1): GroupNorm(32, 512, eps=1e-05, affine=True)
        (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2), groups=32, bias=False)
        (gn2): GroupNorm(32, 512, eps=1e-05, affine=True)
        (conv3): Conv3d(512, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn3): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (relu): PReLU(num_parameters=1)
        (downsample): Sequential(
          (0): Conv3d(512, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
          (1): GroupNorm(32, 1024, eps=1e-05, affine=True)
        )
      )
      (1): ResNeXtDilatedBottleneck(
        (conv1): Conv3d(1024, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn1): GroupNorm(32, 512, eps=1e-05, affine=True)
        (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2), groups=32, bias=False)
        (gn2): GroupNorm(32, 512, eps=1e-05, affine=True)
        (conv3): Conv3d(512, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn3): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (relu): PReLU(num_parameters=1)
      )
      (2): ResNeXtDilatedBottleneck(
        (conv1): Conv3d(1024, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn1): GroupNorm(32, 512, eps=1e-05, affine=True)
        (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2), groups=32, bias=False)
        (gn2): GroupNorm(32, 512, eps=1e-05, affine=True)
        (conv3): Conv3d(512, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn3): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (relu): PReLU(num_parameters=1)
      )
      (3): ResNeXtDilatedBottleneck(
        (conv1): Conv3d(1024, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn1): GroupNorm(32, 512, eps=1e-05, affine=True)
        (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2), groups=32, bias=False)
        (gn2): GroupNorm(32, 512, eps=1e-05, affine=True)
        (conv3): Conv3d(512, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn3): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (relu): PReLU(num_parameters=1)
      )
      (4): ResNeXtDilatedBottleneck(
        (conv1): Conv3d(1024, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn1): GroupNorm(32, 512, eps=1e-05, affine=True)
        (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2), groups=32, bias=False)
        (gn2): GroupNorm(32, 512, eps=1e-05, affine=True)
        (conv3): Conv3d(512, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn3): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (relu): PReLU(num_parameters=1)
      )
      (5): ResNeXtDilatedBottleneck(
        (conv1): Conv3d(1024, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn1): GroupNorm(32, 512, eps=1e-05, affine=True)
        (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2), groups=32, bias=False)
        (gn2): GroupNorm(32, 512, eps=1e-05, affine=True)
        (conv3): Conv3d(512, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn3): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (relu): PReLU(num_parameters=1)
      )
    )
    (layer4): Sequential(
      (0): ResNeXtDilatedBottleneck(
        (conv1): Conv3d(1024, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn1): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (conv2): Conv3d(1024, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2), groups=32, bias=False)
        (gn2): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (conv3): Conv3d(1024, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn3): GroupNorm(32, 2048, eps=1e-05, affine=True)
        (relu): PReLU(num_parameters=1)
        (downsample): Sequential(
          (0): Conv3d(1024, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
          (1): GroupNorm(32, 2048, eps=1e-05, affine=True)
        )
      )
      (1): ResNeXtDilatedBottleneck(
        (conv1): Conv3d(2048, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn1): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (conv2): Conv3d(1024, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2), groups=32, bias=False)
        (gn2): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (conv3): Conv3d(1024, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn3): GroupNorm(32, 2048, eps=1e-05, affine=True)
        (relu): PReLU(num_parameters=1)
      )
      (2): ResNeXtDilatedBottleneck(
        (conv1): Conv3d(2048, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn1): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (conv2): Conv3d(1024, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2), dilation=(2, 2, 2), groups=32, bias=False)
        (gn2): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (conv3): Conv3d(1024, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        (gn3): GroupNorm(32, 2048, eps=1e-05, affine=True)
        (relu): PReLU(num_parameters=1)
      )
    )
  )
  (down4): Sequential(
    (0): Conv3d(2048, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 128, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
  )
  (down3): Sequential(
    (0): Conv3d(1024, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 128, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
  )
  (down2): Sequential(
    (0): Conv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 128, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
  )
  (down1): Sequential(
    (0): Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 128, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
  )
  (fuse1): Sequential(
    (0): Conv3d(512, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 64, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
    (3): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (4): GroupNorm(32, 64, eps=1e-05, affine=True)
    (5): PReLU(num_parameters=1)
    (6): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (7): GroupNorm(32, 64, eps=1e-05, affine=True)
    (8): PReLU(num_parameters=1)
  )
  (attention4): Sequential(
    (0): Conv3d(192, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 64, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
    (3): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (4): GroupNorm(32, 64, eps=1e-05, affine=True)
    (5): PReLU(num_parameters=1)
    (6): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (7): Sigmoid()
  )
  (attention3): Sequential(
    (0): Conv3d(192, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 64, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
    (3): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (4): GroupNorm(32, 64, eps=1e-05, affine=True)
    (5): PReLU(num_parameters=1)
    (6): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (7): Sigmoid()
  )
  (attention2): Sequential(
    (0): Conv3d(192, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 64, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
    (3): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (4): GroupNorm(32, 64, eps=1e-05, affine=True)
    (5): PReLU(num_parameters=1)
    (6): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (7): Sigmoid()
  )
  (attention1): Sequential(
    (0): Conv3d(192, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 64, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
    (3): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (4): GroupNorm(32, 64, eps=1e-05, affine=True)
    (5): PReLU(num_parameters=1)
    (6): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (7): Sigmoid()
  )
  (refine4): Sequential(
    (0): Conv3d(192, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 64, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
    (3): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (4): GroupNorm(32, 64, eps=1e-05, affine=True)
    (5): PReLU(num_parameters=1)
    (6): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (7): GroupNorm(32, 64, eps=1e-05, affine=True)
    (8): PReLU(num_parameters=1)
  )
  (refine3): Sequential(
    (0): Conv3d(192, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 64, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
    (3): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (4): GroupNorm(32, 64, eps=1e-05, affine=True)
    (5): PReLU(num_parameters=1)
    (6): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (7): GroupNorm(32, 64, eps=1e-05, affine=True)
    (8): PReLU(num_parameters=1)
  )
  (refine2): Sequential(
    (0): Conv3d(192, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 64, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
    (3): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (4): GroupNorm(32, 64, eps=1e-05, affine=True)
    (5): PReLU(num_parameters=1)
    (6): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (7): GroupNorm(32, 64, eps=1e-05, affine=True)
    (8): PReLU(num_parameters=1)
  )
  (refine1): Sequential(
    (0): Conv3d(192, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 64, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
    (3): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (4): GroupNorm(32, 64, eps=1e-05, affine=True)
    (5): PReLU(num_parameters=1)
    (6): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (7): GroupNorm(32, 64, eps=1e-05, affine=True)
    (8): PReLU(num_parameters=1)
  )
  (refine): Sequential(
    (0): Conv3d(256, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    (1): GroupNorm(32, 64, eps=1e-05, affine=True)
    (2): PReLU(num_parameters=1)
  )
  (aspp1): ASPP_module(
    (atrous_convolution): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
  )
  (aspp2): ASPP_module(
    (atrous_convolution): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 6, 6), dilation=(1, 6, 6))
    (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
  )
  (aspp3): ASPP_module(
    (atrous_convolution): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 12, 12), dilation=(1, 12, 12))
    (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
  )
  (aspp4): ASPP_module(
    (atrous_convolution): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 18, 18), dilation=(1, 18, 18))
    (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
  )
  (aspp_conv): Conv3d(256, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (aspp_gn): GroupNorm(32, 64, eps=1e-05, affine=True)
  (predict1_4): Conv3d(128, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (predict1_3): Conv3d(128, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (predict1_2): Conv3d(128, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (predict1_1): Conv3d(128, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (predict2_4): Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (predict2_3): Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (predict2_2): Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (predict2_1): Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (predict): Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
)

### for our net
```
[(1,
  Sequential(
  (0): Conv2d(2048, 128, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 128, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
)),
 (2,
  Sequential(
  (0): Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 128, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
)),
 (3,
  Sequential(
  (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 128, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
)),
 (4,
  Sequential(
  (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 128, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
)),
 (5,
  Sequential(
  (0): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 64, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
  (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): GroupNorm(32, 64, eps=1e-05, affine=True)
  (5): PReLU(num_parameters=1)
  (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): GroupNorm(32, 64, eps=1e-05, affine=True)
  (8): PReLU(num_parameters=1)
)),
 (6,
  Sequential(
  (0): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 64, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
  (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): GroupNorm(32, 64, eps=1e-05, affine=True)
  (5): PReLU(num_parameters=1)
  (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): Sigmoid()
)),
 (7,
  Sequential(
  (0): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 64, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
  (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): GroupNorm(32, 64, eps=1e-05, affine=True)
  (5): PReLU(num_parameters=1)
  (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): Sigmoid()
)),
 (8,
  Sequential(
  (0): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 64, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
  (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): GroupNorm(32, 64, eps=1e-05, affine=True)
  (5): PReLU(num_parameters=1)
  (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): Sigmoid()
)),
 (9,
  Sequential(
  (0): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 64, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
  (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): GroupNorm(32, 64, eps=1e-05, affine=True)
  (5): PReLU(num_parameters=1)
  (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): Sigmoid()
)),
 (10,
  Sequential(
  (0): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 64, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
  (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): GroupNorm(32, 64, eps=1e-05, affine=True)
  (5): PReLU(num_parameters=1)
  (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): GroupNorm(32, 64, eps=1e-05, affine=True)
  (8): PReLU(num_parameters=1)
)),
 (11,
  Sequential(
  (0): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 64, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
  (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): GroupNorm(32, 64, eps=1e-05, affine=True)
  (5): PReLU(num_parameters=1)
  (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): GroupNorm(32, 64, eps=1e-05, affine=True)
  (8): PReLU(num_parameters=1)
)),
 (12,
  Sequential(
  (0): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 64, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
  (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): GroupNorm(32, 64, eps=1e-05, affine=True)
  (5): PReLU(num_parameters=1)
  (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): GroupNorm(32, 64, eps=1e-05, affine=True)
  (8): PReLU(num_parameters=1)
)),
 (13,
  Sequential(
  (0): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 64, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
  (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): GroupNorm(32, 64, eps=1e-05, affine=True)
  (5): PReLU(num_parameters=1)
  (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): GroupNorm(32, 64, eps=1e-05, affine=True)
  (8): PReLU(num_parameters=1)
)),
 (14,
  Sequential(
  (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
  (1): GroupNorm(32, 64, eps=1e-05, affine=True)
  (2): PReLU(num_parameters=1)
)),
 (15,
  ASPP_module(
  (atrous_convolution): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
)),
 (16,
  ASPP_module(
  (atrous_convolution): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 6, 6), dilation=(1, 6, 6))
  (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
)),
 (17,
  ASPP_module(
  (atrous_convolution): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 12, 12), dilation=(1, 12, 12))
  (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
)),
 (18,
  ASPP_module(
  (atrous_convolution): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 18, 18), dilation=(1, 18, 18))
  (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
)),
 (19, Conv3d(256, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))),
 (20, GroupNorm(32, 64, eps=1e-05, affine=True)),
 (21, Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))),
 (22, Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))),
 (23, Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))),
 (24, Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))),
 (25, Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))),
 (26, Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))),
 (27, Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))),
 (28, Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))),
 (29, Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))),
 (30,
  AfterRefine(
  (aspp1): ASPP_module_2d(
    (atrous_convolution): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1, 1))
    (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
  )
  (aspp2): ASPP_module_2d(
    (atrous_convolution): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 6, 6), dilation=(1, 6, 6))
    (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
  )
  (aspp3): ASPP_module_2d(
    (atrous_convolution): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 12, 12), dilation=(1, 12, 12))
    (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
  )
  (aspp4): ASPP_module_2d(
    (atrous_convolution): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 18, 18), dilation=(1, 18, 18))
    (group_norm): GroupNorm(32, 64, eps=1e-05, affine=True)
  )
  (aspp_conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
  (aspp_gn): GroupNorm(32, 64, eps=1e-05, affine=True)
  (predict): Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
))]
```
