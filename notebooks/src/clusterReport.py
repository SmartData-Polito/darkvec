import os
import matplotlib.pyplot as plt
import fastplot
import seaborn as sns
from cycler import cycler
from src.callbacks import *

cc = (cycler('color',['k', 'r', 'b', 'g', 'y', 'm', 'c'])+
      cycler('linestyle',['-', '--', '-.', ':', '-', '--', '-.']))

def generate_report(sh_df, filtered, daily, tsne_df, cNAME):
    try:
        os.mkdir('cluster_reports')
    except:
        os.system('rm -rf cluster_reports')
        os.mkdir('cluster_reports')
    os.mkdir('cluster_reports/fig')


    mainfile="""
    \\documentclass[a4paper]{article}

    \\usepackage{extract}
    \\usepackage{fancyhdr}
    \\usepackage{import}
    \\usepackage[latin1]{inputenc}
    \\usepackage{lipsum}
    \\usepackage{amsmath}
    \\usepackage{amsfonts}
    \\usepackage{amssymb}
    \\usepackage{graphicx}
    \\usepackage{listings}
    \\usepackage{float}
    \\usepackage{booktabs}
    \\usepackage{multirow}
    \\usepackage{subcaption}
    \\usepackage[most]{tcolorbox}
    \\usepackage[hmargin=2cm,vmargin=2.5cm]{geometry}
    \\usepackage{lscape}
    \\lstdefinestyle{mystyle}{
        basicstyle=\\ttfamily\\footnotesize,
        breakatwhitespace=false,         
        breaklines=true,                 
        captionpos=b,                    
        keepspaces=true,                 
        numbersep=5pt,                  
        showspaces=false,                
        showstringspaces=false,
        showtabs=false,                  
        tabsize=4
    }

    \\lstset{style=mystyle}

    \\renewcommand{\\title}{\\huge{DarkVec: Clustering Report}}
    \\renewcommand{\\author}{}
    \\newcommand{\\place}{}
    \\renewcommand{\\date}{}
    \\pagestyle{fancy}
    %%%%%%%%%%% HEADER / FOOTER %%%%%%%%%%%
    \\fancyhf{}
    \\lhead{\\title}
    \\rhead{\\rightmark}
    \\cfoot{\\thepage}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%% NEW COMMANDS %%%%%%%%%%%
    \\renewcommand{\\maketitle}[1]{{\\noindent\\huge \\textbf{\\title}}\\par\\vspace{2.5mm}\\author\\par\\vspace{2.5mm}\\place\\par\\vspace{2.5mm}\\date\\vspace{5mm}}
    \\renewcommand{\\sectionmark}[1]{\\markright{\\arabic{section}.\\ #1}}

    \\newcommand{\\R}[2]{$\\mathbb{R}^{#1 \\times #2}$}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    \\begin{document}
    \\thispagestyle{empty}

    \\maketitle
    """
    if cNAME == 'demonstrative': ranges = 7
    else: ranges=sh_df.C.unique().shape[0]
    for C in range(ranges):
        mainfile+="""
        \\include{cluster"""
        mainfile+=str(C)
        mainfile+="""}
        """
        cluster_df = sh_df[sh_df.C == C]
        cluster_trace = filtered[filtered.ip.isin(cluster_df.ip)]
        cluster_daily = daily[daily.ip.isin(cluster_df.ip)]
        tokens ={v:k for k,v in enumerate(cluster_trace.ip.unique())}
        cluster_trace['token'] = cluster_trace.ip.apply(lambda x: tokens[x])
        avg_sh = round(np.mean(cluster_df.sh), 3)
        
        rep = """\\section{Cluster """
        rep += str(C)
        rep += """. Silhouette: """
        rep += str(avg_sh)
        rep += """}

        \\noindent
        \\begin{minipage}{0.6\\textwidth}
        """

        senders = cluster_df.shape[0]
        rep += f"""{senders} distinct senders with the following ground truth classes:\n"""
        rep += """    \\begin{itemize}\n"""

        to_replace = {'mirai-like':'Mirai-like', 'unknown':'Unknown', 'stretchoid':'Stretchoid', 
                    'binaryedge':'Binaryedge', 'ipip':'Ipip', 'shodan':'Shodan', 
                    'engin-umich':'Engin-umich', 'sharashka':'Sharashka', 'internet-census':'Internet-census', 
                    'censys':'Censys', 'shadowserver.org':'Shadowserver', 'alpha strike labs':'AlphaStrike',
                    'bitsight':'Bitsight', 'net systems research':'NetSys', 'cloud system networks':'CSN',}
        classes = cluster_df.value_counts('class').reset_index()
        for i in range(classes.shape[0]):
            _class = to_replace[classes.iloc[i]['class']]
            _senders = classes.iloc[i][0]
            if _senders > 1:
                rep += f"""         \\item {_class}. {_senders} senders\n"""
            else:
                rep += f"""         \\item {_class}. {_senders} sender\n"""
        rep += """    \\end{itemize}\n"""        

        packets = cluster_daily.shape[0]
        packets_perc = round(packets*100/daily.shape[0], 1)

        packets, packets_perc
        rep += f"""{packets} packets sent in the last day. {packets_perc}\% of the last day traffic. """
        if 'mirai-like' in cluster_df['class'].unique():

            mirai_traffic = cluster_daily[cluster_daily.ip.isin(cluster_df[cluster_df['class']=='mirai-like'].ip)].shape[0]
            mirai_traffic = round(mirai_traffic*100/cluster_daily.shape[0], 1)

            rep += f"""{mirai_traffic}\% of cluster traffic has the Mirai fingerprint.\n"""
        else:
            rep += "\n"

        rep += """
        \\end{minipage}
        \\hfill
        \\begin{minipage}{0.4\\textwidth}
        \\begin{figure}[H]
            \\centering
            \\includegraphics[width=\\linewidth]{fig/cluster"""
        rep += str(C)
        rep += """.png}
            \\caption{Cluster """
        rep += str(C)
        rep += """. t-SNE projection}
            \\label{fig:tsne"""
        rep += str(C)
        rep += """}
        \\end{figure}
        \\end{minipage}
        """

        sub24 = cluster_df.value_counts('sub24')
        top24 = sub24[:5]

        rep += """
        \\noindent\n"""
        rep += str(sub24.shape[0])
        rep += """ distinct /24 subnets. The top-5 are:
            \\begin{itemize}
                \item """

        for i in range(top24.shape[0]):
            _class = top24.reset_index().iloc[i]['sub24']
            _senders = top24.reset_index().iloc[i][0]
            if _senders > 1:
                rep += f"""{_class} with {_senders} senders, """
            else:
                rep += f"""{_class} with {_senders} sender """
        rep += "\n"
        rep += """    \\end{itemize}\n"""  

        sub16 = cluster_df.value_counts('sub16')
        top16 = sub16[:5]

        rep += """
        \\noindent\n"""
        rep += str(sub16.shape[0])
        rep += """ distinct /16 subnets. The top-5 are:
            \\begin{itemize}
                \item """

        for i in range(top16.shape[0]):
            _class = top16.reset_index().iloc[i]['sub16']
            _senders = top16.reset_index().iloc[i][0]
            if _senders > 1:
                rep += f"""{_class} with {_senders} senders, """
            else:
                rep += f"""{_class} with {_senders} sender """
        rep += "\n"
        rep += """    \\end{itemize}\n"""  

        pp = cluster_trace.value_counts('pp').shape[0]  

        rep += """
        \\noindent\n"""
        rep += str(pp)
        rep += """ ports contacted. The top-5 are:
            \\begin{itemize}
        """

        top5 = cluster_trace.value_counts('pp')[:5].reset_index()
        for i in range(top5.shape[0]):
            port = top5.iloc[i].pp
            pkts = top5.iloc[i][0]
            pkts_perc = round(pkts*100/cluster_trace.shape[0], 1)
            port_sender = cluster_trace[cluster_trace.pp==port].ip.unique().shape[0]
            port_sender_perc = round(port_sender*100/cluster_trace.ip.unique().shape[0], 1)
            rep += """
            \item """
            rep += str(port)
            rep += """ : """
            rep += str(pkts)
            rep += """  sent packets ("""
            rep += str(pkts_perc)
            rep += """ \% of the monthly cluster traffic.) """
            rep += str(port_sender)
            rep += """  senders contacted the port("""
            rep += str(port_sender_perc)
            rep += """ \% of the cluster senders.)"""
        rep+="""
            \\end{itemize}
        """


        rep +="""
        \\clearpage
        \\begin{figure}[H]
        \\centering
        \\begin{subfigure}[]{.45\\linewidth}
            \\centering
            \\includegraphics[width=\\linewidth]{fig/cluster"""
        rep += str(C)
        rep += """_pattern.png}
            \\caption{Activity pattern}
        \\end{subfigure}
        \\begin{subfigure}[]{.45\\linewidth}
            \\centering
            \\includegraphics[width=\\linewidth]{fig/cluster"""
        rep += str(C)
        rep += """_top4traffic.png}
            \\caption{Top-4 ports traffic}
        \\end{subfigure}
        \\begin{subfigure}[]{.45\\linewidth}
            \\centering
            \\includegraphics[width=\\linewidth]{fig/cluster"""
        rep += str(C)
        rep += """_portheatmap.png}
            \\caption{Port Heatmap}
        \\end{subfigure}
        \\begin{subfigure}[]{.45\\linewidth}
            \\centering
            \\includegraphics[width=\\linewidth]{fig/cluster"""
        rep += str(C)
        rep += """_portpattern.png}
            \\caption{Port pattern}
        \\end{subfigure}
        \\caption{Cluster"""
        rep += str(C)
        rep += """ temporal patterns}
        \\end{figure}"""

        with open(f'cluster_reports/cluster{C}.tex', 'w') as f:
            print(rep, file=f)
            
        
        temp = filtered[filtered.ip.isin(sh_df[sh_df.C == C].ip)]
        ptoken = {v:k for k,v in enumerate(temp.pp.unique())}
        temp['ptoken'] = temp.pp.apply(lambda x: ptoken[x])
        iptoken = {v:k for k,v in enumerate(temp.ip.unique())}
        temp['iptoken'] = temp.ip.apply(lambda x: iptoken[x])
        temp['pkts'] = 1
            
        plot = fastplot.plot(None,  None, mode = 'callback', callback = lambda plt: clusterdensityPlot(plt, tsne_df, C),
                        figsize=(7, 5), fontsize=14, style='latex')
        plot.savefig(f'cluster_reports/fig/cluster{C}.png')
        plot.close()
        
        plot = fastplot.plot(None,  None, mode = 'callback', callback = lambda plt: clusterActivityPattern(plt, cluster_trace, C),
                            figsize=(6, 4.5), fontsize=14, style='latex')
        plot.savefig(f'cluster_reports/fig/cluster{C}_pattern.png')
        plot.close()
        
        plot = fastplot.plot(None,  None, mode = 'callback', callback = lambda plt: Top4Traffic(plt, cluster_trace, top5, C),
                            figsize=(6, 4.5), fontsize=14, style='latex', cycler=cc)
        plot.savefig(f'cluster_reports/fig/cluster{C}_top4traffic.png')
        plot.close()
        
        plot = fastplot.plot(None,  None, mode = 'callback', callback = lambda plt: portHeatmap(plt, temp),
                            figsize=(6, 4.5), fontsize=14, style='latex', cycler=cc)
        plot.savefig(f'cluster_reports/fig/cluster{C}_portheatmap.png')
        plot.close()
        
        plot = fastplot.plot(None,  None, mode = 'callback', callback = lambda plt: portPattern(plt, temp),
                            figsize=(6, 4.5), fontsize=14, style='latex', cycler=cc)
        plot.savefig(f'cluster_reports/fig/cluster{C}_portpattern.png')
        plot.close()

    mainfile+="""
    \\end{document}
    """
    with open(f'cluster_reports/main.tex', 'w') as f:
        print(mainfile, file=f)

    os.system(f'cd cluster_reports && pdflatex main.tex && mv main.pdf ../reports/{cNAME}.pdf && rm -rf ../cluster_reports')