import streamlit as st
import streamlit.components.v1 as components

STYLE = """
<style>

body{
    margin:0;
    padding:0;
    background:#0d1117;
    color:white;
    font-family:Arial, Helvetica, sans-serif;
}

.container{
    padding:10px;
}

.row{
    display:flex;
    gap:20px;
    margin-bottom:20px;
}

.card{

    flex:1;
    background:#161b22;
    border-radius:18px;
    padding:15px;
    box-shadow:0 10px 30px rgba(0,0,0,.35);
    transition:0.25s;
}

.card:hover{

    transform:translateY(-4px);

}

.title{
    font-size:20px;
    font-weight:bold;
    margin-bottom:15px;
    text-align:center;
}

.body{

    font-size:14px;

}

.metric{

    display:flex;

    justify-content:space-between;

    margin:8px 0;

    padding:8px 10px;

    border-radius:8px;

    background:#1f2937;

}

.badge{

    display:inline-block;

    padding:4px 10px;

    border-radius:16px;

    background:#2563eb;

    color:white;

    font-size:12px;

    font-weight:bold;

}

.divider{

    height:1px;

    background:#2d3748;

    margin:15px 0;

}

.pipeline{

    display:flex;
    justify-content:center;
    align-items:center;
    gap:18px;
    margin-bottom:15px;

}

.stage{

    padding:10px 18px;

    border-radius:30px;

    background:#1f2937;

    color:white;

    font-weight:600;

}

.active{

    background:linear-gradient(
        90deg,
        #2563eb,
        #60a5fa
    );

}

.arrow{

    font-size:20px;

    color:#60a5fa;

}

.layer{

    height:24px;

    margin:10px auto;

    border-radius:12px;

    background:linear-gradient(
        90deg,
        #60a5fa,
        #2563eb
    );

    text-align:center;

    line-height:24px;

    font-size:12px;

    font-weight:bold;

}

.container{
    overflow-x:auto;
}

</style>
"""


def render_architecture(config):

    html = ""

    html += "<html>"
    html += STYLE
    html += "<body>"
    html += '<div class="container">'
    vae_cfg = config["latent"]
    diff_cfg = config["diffusion"]
    
    html += render_pipeline()

    html += '<div class="row">'
    #html += render_encoder(config)
    html += render_vae(config)
    html += render_diffusion(config)
    #html += render_decoder(config)
    html += "</div>"

    html += '<div class="row">'
    #html += render_noise_schedule(config)
    #html += render_mmd(config)
    html += "</div>"

    html += "</div>"
    html += "</body>"
    html += "</html>"
    return html

def render_pipeline():

    return """

    <div class="pipeline">

        <div class="stage active">Dataset</div>

        <div class="arrow">➜</div>

        <div class="stage active">Encoder</div>

        <div class="arrow">➜</div>

        <div class="stage active">Latent</div>

        <div class="arrow">➜</div>

        <div class="stage active">Diffusion</div>

        <div class="arrow">➜</div>

        <div class="stage active">Decoder</div>

        <div class="arrow">➜</div>

        <div class="stage active">Synthetic</div>

    </div>

    """

def card(title, body):

    return f"""
    <div class="card">

        <div class="title">

            {title}

        </div>

        <div class="body">

            {body}

        </div>

    </div>
    """

def badge(text, color="#3b82f6"):

    return f"""
    <span class="badge"
        style="background:{color};">
        {text}
    </span>
    """

def divider():

    return """
    <div class="divider"></div>
    """

def metric(name, value):

    return f"""
    <div class="metric">

        <span>{name}</span>

        <strong>{value}</strong>

    </div>
    """

def funnel_html(layers):

    maximum = max(layers)

    html = ""

    for layer in layers:

        width = layer / maximum

        html += f"""

        <div
            class="layer"
            style="width:{width*100:.1f}%">

            {layer}

        </div>

        """

    return html

def render_encoder(vae_cfg):

    latent = vae_cfg["latent"]

    if latent["mode"] == "adaptive":

        encoder = ["Auto"]

    else:

        encoder = latent["encoder_dims"]

    # Add input dimension if available
    input_dim = vae_cfg.get("input_dim")

    if input_dim is not None and encoder != ["Auto"]:
        layers = [input_dim] + encoder
    else:
        layers = encoder

    body = ""

    if encoder == ["Auto"]:

        body += badge("Adaptive", "#16a34a")

        body += metric(
            "Heuristic",
            latent["heuristic"].title()
        )

        body += """
        <div style="margin-top:20px;
                    text-align:center;
                    color:#9ca3af;">
            Encoder generated automatically
        </div>
        """

    else:

        body += badge("Custom", "#2563eb")

        body += funnel_html(layers)

        body += divider()

        body += metric(
            "Layers",
            len(encoder)
        )

        body += metric(
            "Latent",
            latent["latent_dim"]
        )

    return card(
        "Encoder",
        body
    )

def funnel_html(layers):

    numeric = [
        x for x in layers
        if isinstance(x, (int, float))
    ]

    if len(numeric) == 0:
        return ""

    maximum = max(numeric)

    html = ""

    for layer in layers:

        if not isinstance(layer, (int, float)):
            continue

        width = max(
            12,
            int((layer / maximum) * 100)
        )

        html += f"""

        <div
            class="layer"
            style="width:{width}%">

            {layer:,}

        </div>

        """

    return html

def render_latent(config):

    latent = config["latent"]

    if latent["mode"] == "adaptive":

        subtitle = f"""
        Strategy : {latent["heuristic"].title()}
        """

    else:

        subtitle = f"""
        Latent : {latent["latent_dim"]}<br>
        Encoder : {" → ".join(map(str, latent["encoder_dims"]))}<br>
        Decoder : {" → ".join(map(str, latent["decoder_dims"]))}
        """

    return f"""
    <div class="card latent">
        <h3>MMD-VAE</h3>
        <p>{subtitle}</p>
    </div>

    <div class="arrow">→</div>
    """

def render_diffusion(config):

    diff = config["diffusion"]
    latent = config["latent"]

    schedule = diff["noise_schedule"]["type"].title()
    arch = diff["architecture"].title()

    # If the user selects "Adaptive", we hide the network visual to prevent crashes
    if latent["mode"] == "adaptive":
        
        body = f"""
        {badge("Adaptive", "#16a34a")}
        <div style="
            margin-top:30px;
            text-align:center;
            color:#9ca3af;
            font-size:16px;
        ">
            Denoising network generated automatically
        </div>
        """
        
    else:
        
        # Latent Diffusion maps the latent space back to itself.
        # We start with the latent_dim, add the hidden layers, and end with the latent_dim.
        latent_dim = latent["latent_dim"]
        layers = [latent_dim] + diff["hidden_dims"] + [latent_dim]
        
        body = f"""
        <div style="
            display:flex;
            justify-content:center;
            align-items:flex-start;
        ">
            {render_network(
                layers,
                "Denoising Network",
                "#a855f7"  # A nice purple color to distinguish it from the autoencoder
            )}
        </div>

        <div style="
            margin-top:30px;
        ">
            {divider()}
            {metric("Architecture", arch)}
            {metric("Timesteps", diff["num_diffusion_steps"])}
            {metric("Noise Schedule", schedule)}
            {metric("Time Conditioning", diff["time_conditioning"]["method"].title())}
        </div>
        """

    return card("Latent Diffusion", body)

def render_decoder(vae_cfg):

    latent = vae_cfg["latent"]

    if latent["mode"] == "adaptive":

        decoder = ["Auto"]

    else:

        decoder = latent["decoder_dims"]

    # Add output dimension if available
    output_dim = vae_cfg.get("input_dim")

    if output_dim is not None and decoder != ["Auto"]:
        layers = decoder + [output_dim]
    else:
        layers = decoder

    body = ""

    if decoder == ["Auto"]:

        body += badge("Adaptive", "#16a34a")

        body += metric(
            "Heuristic",
            latent["heuristic"].title()
        )

        body += """
        <div style="margin-top:20px;
                    text-align:center;
                    color:#9ca3af;">
            Decoder generated automatically
        </div>
        """

    else:

        body += badge("Custom", "#2563eb")

        body += funnel_html(layers)

        body += divider()

        body += metric(
            "Layers",
            len(decoder)
        )

        body += metric(
            "Output",
            output_dim if output_dim is not None else "Unknown"
        )

    return card(
        "Decoder",
        body
    )


def render_network(layers, title, color="#3b82f6"):
    """
    layers : list[int]
        Example:
        Encoder : [18917, 2048, 1024, 512, 256]
        Decoder : [256, 512, 1024, 2048, 18917]
    """

    MAX_NEURONS = 8

    html = f"""
    <div style="
        flex:1;
        display:flex;
        flex-direction:column;
        align-items:center;
    ">

        <div style="
            font-size:22px;
            font-weight:700;
            margin-bottom:20px;
            color:white;
        ">
            {title}
        </div>

        <div style="
            display:flex;
            align-items:center;
            justify-content:center;
            gap:18px;
        ">
    """

    for i, size in enumerate(layers):

        neurons = min(MAX_NEURONS, max(3, int(round(size**0.25))))

        html += f"""
        <div style="
            display:flex;
            flex-direction:column;
            align-items:center;
        ">

            <div style="
                color:#cbd5e1;
                font-size:14px;
                margin-bottom:10px;
                font-weight:600;
            ">
                {size}
            </div>
        """

        for _ in range(neurons):

            html += f"""
            <div style="
                width:16px;
                height:16px;
                border-radius:50%;
                background:{color};
                margin:3px;
                box-shadow:0 0 8px {color};
            "></div>
            """

        if size > neurons:

            html += """
            <div style="
                color:#94a3b8;
                font-size:18px;
                line-height:12px;
            ">
            ⋮
            </div>
            """

        html += "</div>"

        if i != len(layers)-1:

            html += """
            <div style="
                font-size:28px;
                color:#64748b;
                padding:0 8px;
            ">
                →
            </div>
            """

    html += """
        </div>
    </div>
    """

    return html

def render_vae(config):

    latent = config["latent"]
    vae_params = config.get("vae_params", {})

    if latent["mode"] == "adaptive":

        return card(
            "MMD-VAE",
            f"""
            {badge("Adaptive","#16a34a")}

            {metric("Heuristic",
                    latent["heuristic"].title())}

            <div style="
                margin-top:30px;
                text-align:center;
                color:#9ca3af;
                font-size:16px;
            ">
                Encoder/Decoder generated automatically
            </div>
            """
        )

    input_dim = config["input_dim"]

    # Calculate layers
    encoder = latent["encoder_dims"] + [latent["latent_dim"]]
    decoder = latent["decoder_dims"]

    body = f"""

    <div style="
        display:flex;
        gap:20px;
        align-items:flex-start;
        justify-content:center;
    ">

        {render_network(
            encoder,
            "Encoder",
            "#3b82f6"
        )}

        <div style="
            font-size:28px;
            color:#64748b;
            padding:0 8px;
            margin-top: 135px; 
        ">
            →
        </div>

        {render_network(
            decoder,
            "Decoder",
            "#f97316"
        )}

    </div>

    <div style="margin-top:30px;">
        {divider()}
        
        <!-- Splitting the parameters into two clean columns -->
        <div style="display: flex; gap: 20px;">
            <div style="flex: 1;">
                <div style="color:#94a3b8; margin-bottom: 8px; font-weight: bold; font-size: 12px; text-transform: uppercase;">
                    Architecture Specs
                </div>
                {metric("Latent Dimension", latent["latent_dim"])}
                {metric("MMD Strategy", config["mmd"]["strategy"].title())}
                {metric("Activation", vae_params.get("activation", "N/A"))}
                {metric("LayerNorm", "Yes" if vae_params.get("layernorm") else "No")}
                {metric("Dropout", vae_params.get("dropout", "N/A"))}
            </div>
            
            <div style="flex: 1;">
                <div style="color:#94a3b8; margin-bottom: 8px; font-weight: bold; font-size: 12px; text-transform: uppercase;">
                    Training Specs
                </div>
                {metric("Epochs", vae_params.get("epochs", "N/A"))}
                {metric("Batch Size", vae_params.get("batch_size", "N/A"))}
                {metric("Learning Rate", vae_params.get("learning_rate", "N/A"))}
            </div>
        </div>
    </div>

    """

    return card("MMD-VAE", body)
