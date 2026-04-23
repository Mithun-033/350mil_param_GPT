import torch
import torch.nn as nn
import HybridOptimizer from Muon
import json

criterion=nn.CrossEntropyLoss()
optimizer=HybridOptimizer(model.parameters())

total_steps=1_000_000_000//(512*Config.cwl)
epochs=1
warmup_steps=int(total_steps*0.05)

cosine_steps=int(total_steps-warmup_steps)

warmup_scheduler=torch.optim.lr_scheduler.LinearLR(optimizer.adamw,start_factor=0.2,end_factor=1,total_iters=warmup_steps)
cosine_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.adamw,T_max=cosine_steps,eta_min=Config.lr*0.1)
scheduler=torch.optim.lr_scheduler.SequentialLR(optimizer.adamw,[warmup_scheduler,cosine_scheduler],[warmup_steps])
#TODO try diff scheduler 
warmup_scheduler2=torch.optim.lr_scheduler.LinearLR(optimizer.muon,start_factor=0.2,end_factor=1,total_iters=warmup_steps)
cosine_scheduler2=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.muon,T_max=cosine_steps,eta_min=Config.lr*0.1)
scheduler2=torch.optim.lr_scheduler.SequentialLR(optimizer.muon,[warmup_scheduler2,cosine_scheduler2],[warmup_steps])

val_ids=np.load("none",mmap_mode="r")
val_loss=[]
train_loss=[]
ids=np.load("none",mmap_mode="r")
steps=0

for i in range(epochs):
    start=time.time()
    loss_accum=0
    optimizer.zero_grad()
    model.train()

    for x,y in generator(ids,Config.b_size,Config.cwl,dev):

        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            out=model(x)
            out=out.view(-1,Config.vocab_size)
            y=y.view(-1)
            loss=criterion(out,y)

        loss=loss/Config.grad_steps
        loss_accum+=loss.item()

        loss.backward()
        steps+=1

        if steps%Config.grad_steps==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            scheduler.step()
            scheduler2.step()
            optimizer.zero_grad()

            if steps%(20*Config.grad_steps)==0:
                tokens_processed=20*Config.grad_steps*Config.b_size*Config.cwl
                end=time.time()
                time_taken=end-start
                tps=tokens_processed/time_taken
                print(f"Loss:{loss_accum/20:.5f} | Time:{time_taken:.2f}s | Tokens/s:{tps:.0f}")
                train_loss.append(loss_accum/20)
                loss_accum=0
                start=end

        if steps%(Config.grad_steps*100)==0:
            torch.save({
                "model":model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "step":steps
            },"bf16-proper-4.pt")

            model.eval()

            loss_accum_val=0
            val_steps=0

            with torch.no_grad():
                for x,y in generator(val_ids,Config.b_size,Config.cwl,dev):

                    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                        out=model(x)
                        out=out.view(-1,Config.vocab_size)
                        y=y.view(-1)
                        loss=criterion(out,y)

                    loss_accum_val+=loss.item()
                    val_steps+=1

            val_loss.append(loss_accum_val/val_steps)

            with open("bf16-4.json","w") as f:
                json.dump(val_loss,f)

            print(f"bf16:- Val Loss :{loss_accum_val/val_steps:.5f} Count:{len(val_loss)}")

            model.train()

torch.save(model.state_dict(),"GPT-bf16-3.pt")

plt.plot(val_loss,label="Validation Loss")
plt.plot(train_loss,label="Train Loss")
plt.title("Train vs Validation Loss Curve")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve_bf16.png")
plt.show()
