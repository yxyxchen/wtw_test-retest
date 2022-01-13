library('ggplot2')
themeColor = '#B72627'
expName = "Exp.1"
conditionColors = c("#1b7837", "#762a83")
conditionColorBacks = c( "#a6dba0", "#c2a5cf")
# annotations for p values 
symnum.args <- list(cutpoints = c(0,0.001, 0.01, 0.05, 1),
                    symbols = c("***", "**", "*", "ns"))
# plot theme
myTheme = theme( panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), text=element_text(face = "bold", size = 15),
        axis.line = element_line(size = 1),
        panel.background = element_rect(fill = "white"),
        strip.background = element_blank(),
        # adjust X-axis labels; also adjust their position using margin (acts like a bounding box)
        # using margin was needed because of the inwards placement of ticks
        axis.text.x = element_text(margin = unit(c(t = 2.5, r = 0, b = 0, l = 0), "mm")),
        # adjust Y-axis title
        axis.title.y = element_text(face = "bold"),
        # adjust Y-axis labels
        axis.text.y = element_text(margin = unit(c(t = 0, r = 2.5, b = 0, l = 0), "mm")),
        # length of tick marks - negative sign places ticks inwards
        axis.ticks.length = unit(-1.4, "mm"),
        # width of tick marks in mm
        axis.ticks = element_line(size = 1)
        ) 
