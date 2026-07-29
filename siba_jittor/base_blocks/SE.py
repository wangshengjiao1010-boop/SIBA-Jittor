from jittor import nn
import jittor as jt

'''
Squeeze-and-Excitation Networks
'''
class se_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(se_module, self).__init__()
        assert channel >= reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction), bias=False),
            nn.ReLU(),
            nn.Linear(int(channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def execute(self, x): 
        b, c, _, _ = x.shape
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y)        
        y = y.view(b, c, 1, 1)
        return x * y
    

if __name__ == '__main__':
    temp = jt.randn((1,32,64,64))
    model = se_module(32)
    out = model(temp)
    print(out.shape)
