import torch
import collections

SingleTrackParamsList = ['m', 'g', 'I_z', 'L', 'lr', 'lf', 'Cd0', 'Cd2', 'Cd1',
                         'mu_static', 'eps', 'CoM_height', "K_lt", "vxtau", "vxk", "vytau", "vyk", "rtau", "rk", "Cl0", "Cl1", "Cl2"]
SingleTrackParams = collections.namedtuple('SingleTrackParams', SingleTrackParamsList,
                                           defaults=(None,) * len(SingleTrackParamsList))

class DefaultSingleTrackLoadTransferParameters(torch.nn.Module):
    def __init__(self) -> None:
        super(DefaultSingleTrackLoadTransferParameters, self).__init__()
        # Define constant tensors for detached parameters
        self.m_ = torch.tensor([5.1], requires_grad=False)
        self.g_ = torch.tensor([9.81], requires_grad=False)
        self.L_ = torch.tensor([0.33], requires_grad=False)
        self.mu_static_ = torch.tensor([0.8], requires_grad=False)
        self.eps_ = torch.tensor([1e-6], requires_grad=False)
        
        self.register_buffer('m', self.m_)
        self.register_buffer('g', self.g_)
        self.register_buffer('L', self.L_)
        self.register_buffer('mu_static', self.mu_static_)
        self.register_buffer('eps', self.eps_)        

        self.log_I_z = torch.nn.Parameter(torch.log(torch.tensor([0.46])))
        self.log_lr = torch.nn.Parameter(torch.log(torch.tensor([0.115])))
        self.log_Cd0 = torch.nn.Parameter(torch.log(torch.tensor([0.01])))
        self.log_Cd1 = torch.nn.Parameter(torch.log(torch.tensor([0.01])))
        self.log_Cd2 = torch.nn.Parameter(torch.log(torch.tensor([0.01])))
        self.log_CoM_height = torch.nn.Parameter(torch.log(torch.tensor([0.05])))
        self.log_K_lt = torch.nn.Parameter(torch.log(torch.tensor([1.0])))
        self.log_vxtau = torch.nn.Parameter(torch.log(torch.tensor([0.01])))
        self.log_vxk = torch.nn.Parameter(torch.log(torch.tensor([1.0])))
        self.log_vytau = torch.nn.Parameter(torch.log(torch.tensor([0.01])))
        self.log_vyk = torch.nn.Parameter(torch.log(torch.tensor([1.0])))
        self.log_rtau = torch.nn.Parameter(torch.log(torch.tensor([0.01])))
        self.log_rk = torch.nn.Parameter(torch.log(torch.tensor([1.0])))
        self.Cl0 = torch.nn.Parameter(torch.tensor([0.0]))
        self.Cl1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.Cl2 = torch.nn.Parameter(torch.tensor([0.0]))
        

    def forward(self) -> SingleTrackParams:
        return SingleTrackParams(
            m=self.m,
            g=self.g,
            I_z=self.log_I_z.exp(),
            L=self.L,
            lr=self.log_lr.exp(),
            lf=self.L - self.log_lr.exp(),  # lf = L - lr
            Cd0=self.log_Cd0.exp(),
            Cd2=self.log_Cd2.exp(),
            Cd1=self.log_Cd1.exp(),
            mu_static=self.mu_static,
            eps=self.eps,
            CoM_height=self.log_CoM_height.exp(),
            K_lt=self.log_K_lt.exp(),
            vxtau=self.log_vxtau.exp(),
            vxk=self.log_vxk.exp(),
            vytau=self.log_vytau.exp(),
            vyk=self.log_vyk.exp(),
            rtau=self.log_rtau.exp(),
            rk=self.log_rk.exp(),
            Cl0=self.Cl0,
            Cl1=self.Cl1,
            Cl2=self.Cl2,
        )
    
    def get_parameters_vector(self):
        return torch.cat([
            self.log_I_z.unsqueeze(0).exp(),
            self.log_lr.unsqueeze(0).exp(),
            self.log_Cd0.unsqueeze(0).exp(),
            self.log_Cd1.unsqueeze(0).exp(),
            self.log_Cd2.unsqueeze(0).exp(),
            self.log_CoM_height.unsqueeze(0).exp(),
            self.log_K_lt.unsqueeze(0).exp(),
            self.log_vxtau.unsqueeze(0).exp(),
            self.log_vxk.unsqueeze(0).exp(),
            self.log_vytau.unsqueeze(0).exp(),
            self.log_vyk.unsqueeze(0).exp(),
            self.log_rtau.unsqueeze(0).exp(),
            self.log_rk.unsqueeze(0).exp(),
            self.Cl0.unsqueeze(0),
            self.Cl1.unsqueeze(0),
            self.Cl2.unsqueeze(0),
        ], dim=0)