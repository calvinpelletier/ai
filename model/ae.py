import ai.model as m


class ImgAutoencoder(m.Model):
    '''Image autoencoder.

    INPUT
        tensor[b, c, h, w]

    OUTPUT
        tensor[b, c, h, w]
            range: [-1, 1]

    ARGS
        imsize : int
            input image size
        bottleneck : int
            smallest feature map size
        nc_min : int
            initially deepen the image to <nc_min> channels
        nc_max : int
            maximum number of channels for internal feature maps
        enc_block : callable
            factory that produces blocks for the encoder.
            enc_block(size, nc1, nc2)
                size : int
                    size of the input feature map to the block
                nc1 : int
                    number of input channels
                nc2 : int
                    number of output channels
        dec_block : callable
            same as enc_block but for the decoder
    '''

    def __init__(s,
        imsize: int,
        bottleneck: int,
        nc_min: int = 32,
        nc_max: int = 512,
        enc_block=lambda size, nc1, nc2: m.resblk(nc1, nc2, stride=2),
        dec_block=lambda size, nc1, nc2: m.resblk(nc1, nc2, stride=.5),
    ):
        super().__init__()

        s.encode = m.seq(
            m.conv(3, nc_min, actv='mish'),
            m.pyramid(imsize, bottleneck, nc_min, nc_max, enc_block),
        )

        s.decode = m.seq(
            m.pyramid(bottleneck, imsize, nc_min, nc_max, dec_block),
            m.conv(nc_min, 3, actv='tanh'),
        )

    def forward(s, x):
        return s.decode(s.encode(x))
