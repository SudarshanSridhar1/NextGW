import torch as tr
import torch.nn as nn

from _03_graph_construction import graph_data, baseGNN
from _05_gnn_trainer import train

data, ptdf, rhs_all, F_max_branches, Pf_base = graph_data("texas", cluster=True)

model = baseGNN(hidden_channels=64, edge_dim=data["bus", "wire", "bus"].edge_attr.shape[1])
model.set_constraints(ptdf, rhs_all, F_max_branches, Pf_base)

criterion = nn.MSELoss()
optimizer = tr.optim.Adam(model.parameters(), lr=0.01)
train_mask = tr.rand(data["bus"].x.size(0)) < 0.8
test_mask = ~train_mask

name = "Train Test"

print("Starting training...")

trainer = train(
    model = model, data = data, 
    criterion = criterion, optimizer = optimizer, 
    train_mask = train_mask, test_mask = test_mask,
    device='cpu', save_location=None,
    name=name, gradient_location=None, printinfo=True
)

trainer.set_constraints(ptdf, rhs_all)

lr_scheduler = tr.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
#resume_state = r"ml_pipeline\outputs\checkpoint.pth"
resume_state = None

tchart, vchart, model = trainer.train_loop(
                epochs=100000, lr_scheduler=lr_scheduler, save_increment=None,
                time_limit=None, lr_decay_warm_restarts=0.98, clip_grad_norm=0.25,
                save_best=True, predict_per_epoch=None, resume_state=resume_state,
                file_path=r"ml_pipeline\outputs\loss_graphs", dist=10, penalty_multi = 100
)

#Ideal MSE in my eyes for v0: ~25 or something like that 
#Average prediction is a few hundred MW

trainer.print_model_stats(output = "full")
trainer.count_parameters()
outputs = trainer.predict(mask = test_mask)

print(outputs[test_mask].squeeze())
print(data["bus"].y[test_mask].squeeze())
print(outputs[test_mask].squeeze() - data["bus"].y[test_mask].squeeze())