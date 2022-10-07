import os
import numpy as np
from shapely.geometry import Point, Polygon
import constants

from remi import App, gui


class VoronoiApp(App):
    def __init__(self, *args):
        res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
        super(VoronoiApp, self).__init__(*args, static_file_path={'res': res_path})

    def convert_coord(self, coord):
        if not isinstance(coord, Point):
            coord = Point(coord)
        scale = min(self.scale.x * self.vis_width, self.scale.y * self.vis_height)
        # coord = coord.translate(self.translate.x, self.translate.y)
        # coord = coord.scale(x=scale, y=scale)
        # coord = coord.translate((self.vis_width * self.padding_factor) / 2, (self.vis_height * self.padding_factor) / 2)
        # return coord
        off_x = (self.vis_width * self.padding_factor) / 2
        off_y = (self.vis_height * self.padding_factor) / 2
        return Point((coord.x + self.translate.x) * scale + off_x, (coord.y + self.translate.y) * scale + off_y )

    def draw_polygon(self, poly):
        svg_poly = gui.SvgPolygon(len(poly.exterior.coords))
        for p in poly.exterior.coords:
            p = self.convert_coord(p)
            svg_poly.add_coord(float(p.x), float(p.y))
        return svg_poly

    def draw_point(self, point):
        point = self.convert_coord(point)
        return gui.SvgCircle(float(point.x), float(point.y), 1.0)

    def draw_line(self, a, b):
        point1 = self.convert_coord(a)
        point2 = self.convert_coord(b)
        return gui.SvgLine(float(point1.x), float(point1.y), float(point2.x), float(point2.y))

    def draw_circle(self, center, radius):
        scale = min(self.scale.x * self.vis_width, self.scale.y * self.vis_height)
        center = self.convert_coord(center)
        radius = scale * radius
        return gui.SvgCircle(float(center.x), float(center.y), float(radius))

    def draw_text(self, point, text):
        point = self.convert_coord(point)
        return gui.SvgText(float(point.x), float(point.y), text)

    def main(self, *userdata):
        self.voronoi_game, self.logger = userdata
        self.voronoi_game.set_app(self)
        self.vis_width = constants.vis_width
        self.vis_height = constants.vis_height
        self.padding_factor = 1. + 2 * constants.vis_padding

        self.prev_day = 0
        self.prev_state = 0
        self.curr_day = 0
        self.curr_state = 0

        mainContainer = gui.Container(
            style={'width': '100%', 'height': '100%', 'overflow': 'auto', 'text-align': 'center'})
        mainContainer.style['justify-content'] = 'center'
        mainContainer.style['align-items'] = 'center'
        mainContainer.set_layout_orientation(gui.Container.LAYOUT_HORIZONTAL)

        boardContainer = gui.Container(
            style={'width': '55%', 'height': '100%', 'overflow': 'auto', 'text-align': 'center'})
        boardContainer.style['justify-content'] = 'center'
        boardContainer.style['align-items'] = 'center'

        menuContainer = gui.Container(
            style={'width': '45%', 'height': '100%', 'overflow': 'auto', 'text-align': 'center'})
        menuContainer.style['justify-content'] = 'center'
        menuContainer.style['align-items'] = 'center'

        header_label = gui.Label(
            "Project 2: Voronoi - (t = {}, n = {})".format(self.voronoi_game.last_day, self.voronoi_game.spawn_day),
            style={'font-size': '36px', 'font-weight': 'bold'})
        menuContainer.append(header_label)

        bt_hbox = gui.HBox()
        go_start_bt = gui.Button("Back to Start")
        prev_day_bt = gui.Button("Previous Day")
        prev_state_bt = gui.Button("Previous State")
        next_state_bt = gui.Button("Next State")
        next_day_bt = gui.Button("Next Day")
        go_end_bt = gui.Button("Skip to End")

        bt_hbox.append([go_start_bt, prev_day_bt, prev_state_bt, next_state_bt, next_day_bt, go_end_bt])

        go_start_bt.onclick.do(self.go_start_bt_press)
        prev_day_bt.onclick.do(self.prev_day_bt_press)
        prev_state_bt.onclick.do(self.prev_state_bt_press)
        next_state_bt.onclick.do(self.next_state_bt_press)
        next_day_bt.onclick.do(self.next_day_bt_press)
        go_end_bt.onclick.do(self.go_end_bt_press)

        menuContainer.append(bt_hbox)

        day_label = gui.Label("Press a button or choose a day: ", style={'margin': '5px'})

        ch_hbox = gui.HBox()
        self.view_drop_down = gui.DropDown(style={'padding': '5px', 'text-align': 'center'})
        for i in range(self.voronoi_game.last_day):
            self.view_drop_down.append("Day {}".format(i + 1), i)

        self.view_drop_down.onchange.do(self.view_drop_down_changed)

        self.labels = []
        self.labels.append(gui.Label("Start of Day", style={'margin': '5px auto'}))

        ch_hbox.append([day_label, self.view_drop_down, self.labels[0]])

        menuContainer.append(gui.Label())
        menuContainer.append(ch_hbox)

        lb_hbox = [gui.HBox()]
        name_label = gui.Label("Name of the Player", style={'margin': '5px auto', 'font-weight': 'bold'})
        score_label = gui.Label("Cells Currently Occupied", style={'margin': '5px auto', 'font-weight': 'bold'})
        total_label = gui.Label("Total Score", style={'margin': '5px auto', 'font-weight': 'bold'})
        lb_hbox[0].append([name_label, score_label, total_label])
        menuContainer.append(gui.Label())
        menuContainer.append(lb_hbox[0])

        for i in range(constants.no_of_players):
            lb_hbox.append(gui.HBox(style={'background-color': constants.tile_color[i]}))
            self.labels.append(gui.Label("{}".format(self.voronoi_game.player_names[i]), style={'margin': '5px auto'}))
            self.labels.append(gui.Label("{}".format(self.voronoi_game.player_score[0][0][i]),
                                         style={'margin': '5px auto'}))
            self.labels.append(gui.Label("{}".format(0), style={'margin': '5px auto'}))
            lb_hbox[i + 1].append(self.labels[((i * 3) + 1):((i * 3) + 4)])
            menuContainer.append(lb_hbox[i + 1])

        self.load_map()
        self.svgplot = gui.Svg(width="{}vw".format(constants.vis_width_ratio * self.padding_factor),
                               height="{}vh".format(100 * constants.vis_height_ratio * self.padding_factor),
                               style={'background-color': '#FFFFFF', 'margin': '0 auto',
                                      'min-width': str(self.vis_width * self.padding_factor),
                                      'min-height': str(self.vis_height * self.padding_factor)})
        self.svgplot.set_viewbox(0, 0, self.vis_width * self.padding_factor, self.vis_height * self.padding_factor)
        self.svgplot.attr_preserveAspectRatio = "xMidYMid"
        self.plot_base()
        self.base_keys = list(self.svgplot.children.keys())
        self.display_map(0, 0)

        boardContainer.append(self.svgplot)
        mainContainer.append(boardContainer)
        mainContainer.append(menuContainer)

        return mainContainer

    def load_map(self):
        self.base = [Point(constants.base[0]), Point(constants.base[1]),
                     Point(constants.base[2]), Point(constants.base[3])]
        self.voronoi_map = Polygon([[0, 0], [0, 100], [100, 100], [100, 0], [0, 0]])
        self.translate = Point(-(constants.min_map_dim + constants.max_map_dim) / 2,
                                                -(constants.min_map_dim + constants.max_map_dim) / 2)
        self.scale = Point(1 / (constants.max_map_dim - constants.min_map_dim + 1),
                                            1 / (constants.max_map_dim - constants.min_map_dim + 1))

        self.logger.info(
            "Translating visualization by x={}, y={}".format(float(self.translate.x), float(self.translate.y)))
        self.logger.info("Base Scaling visualization by factors {}".format(float(self.scale.x), float(self.scale.y)))

    def plot_base(self):
        self.reset_svgplot()
        unit_h = [[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]
        for i in range(constants.no_of_players):
            base_off_x = self.base[i].x - 1
            base_off_y = self.base[i].y - 1
            base_marker = Polygon([(x + base_off_x, y + base_off_y) for x, y in unit_h])

            t = self.draw_polygon(base_marker)
            t.set_fill(constants.player_color[i])
            self.svgplot.append(t)

        p = self.draw_polygon(self.voronoi_map)
        p.set_stroke(1, "black")
        self.svgplot.append(p)

    def display_map(self, day, state):
        self.prev_day = self.curr_day
        self.prev_state = self.curr_state
        self.curr_day = day
        self.curr_state = state

        self.reset_svgplot()
        self.plot_tiles()
        self.base_keys = list(self.svgplot.children.keys())
        self.update_table()
        self.plot_units()

        self.view_drop_down.select_by_key(day)

        if state == 0:
            self.set_label_text("Start of Day")
        elif state == 1:
            self.set_label_text("Movement During the Day")
        else:
            self.set_label_text("End of Day")

    def reset_svgplot(self):
        if len(self.svgplot.children) == 0:
            self.svgplot.empty()
        else:
            for k in list(self.svgplot.children.keys()):
                if k not in self.base_keys:
                    self.svgplot.remove_child(self.svgplot.children[k])

    def plot_tiles(self):
        if self.curr_day == 0 and self.curr_state == 0:
            tmp = [Point(25, 25), Point(25, 75),
                   Point(75, 75), Point(75, 25)]
            q_t = [[0, 0], [0, 50], [50, 50], [50, 0], [0, 0]]
            for i in range(constants.no_of_players):
                off_x = tmp[i].x - 25
                off_y = tmp[i].y - 25
                base_marker = Polygon([(x + off_x, y + off_y) for x, y in q_t])

                t = self.draw_polygon(base_marker)
                t.set_fill(constants.tile_color[i])
                self.svgplot.append(t)

        else:
            unit_t = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
            for i in range(constants.max_map_dim):
                for j in range(constants.max_map_dim):
                    if self.voronoi_game.map_states[self.curr_day][self.curr_state][i][j] != \
                            self.voronoi_game.map_states[self.prev_day][self.prev_state][i][j]:
                        tile = Polygon([(x + i, y + j) for x, y in unit_t])
                        t = self.draw_polygon(tile)
                        if self.voronoi_game.map_states[self.curr_day][self.curr_state][i][j] > 0:
                            t.set_fill(constants.tile_color[
                                           self.voronoi_game.map_states[self.curr_day][self.curr_state][i][j] - 1])
                        else:
                            t.set_fill(constants.dispute_color)
                        self.svgplot.append(t)

    def plot_units(self):
        for i in range(constants.no_of_players):
            for j in range(len(self.voronoi_game.unit_id[self.curr_day][self.curr_state][i])):
                c = self.draw_circle(self.voronoi_game.unit_pos[self.curr_day][self.curr_state][i][j], 0.5)
                c.set_fill(constants.player_color[i])
                c.set_stroke(1, "black")
                self.svgplot.append(c)

                text = self.draw_text(self.voronoi_game.unit_pos[self.curr_day][self.curr_state][i][j],
                                      self.voronoi_game.unit_id[self.curr_day][self.curr_state][i][j])
                text.set_stroke(1, "black")
                text.set_style(style="font-size:10")
                self.svgplot.append(text)

                if self.curr_state == 1:
                    path = self.draw_line(self.voronoi_game.unit_pos[self.curr_day][self.curr_state - 1][i][j],
                                          self.voronoi_game.unit_pos[self.curr_day][self.curr_state][i][j])
                    path.set_stroke(1, "black")
                    self.svgplot.append(path)

    def set_label_text(self, text, label_num=0):
        self.labels[label_num].set_text(text)

    def go_start_bt_press(self, widget):
        self.set_label_text("Processing...")
        self.do_gui_update()
        self.display_map(0, 0)

    def prev_day_bt_press(self, widget):
        if self.curr_day != 0:
            self.set_label_text("Processing...")
            self.do_gui_update()
            self.display_map(self.curr_day - 1, 0)

    def prev_state_bt_press(self, widget):
        if self.curr_state == 0:
            if self.curr_day != 0:
                self.set_label_text("Processing...")
                self.do_gui_update()
                self.display_map(self.curr_day - 1, 2)
        else:
            self.set_label_text("Processing...")
            self.do_gui_update()
            self.display_map(self.curr_day, self.curr_state - 1)

    def next_state_bt_press(self, widget):
        if self.curr_state == 2:
            if self.curr_day != self.voronoi_game.last_day - 1:
                self.set_label_text("Processing...")
                self.do_gui_update()
                self.display_map(self.curr_day + 1, 0)
        else:
            self.set_label_text("Processing...")
            self.do_gui_update()
            self.display_map(self.curr_day, self.curr_state + 1)

    def next_day_bt_press(self, widget):
        if self.curr_day != self.voronoi_game.last_day - 1:
            self.set_label_text("Processing...")
            self.do_gui_update()
            self.display_map(self.curr_day + 1, 0)

    def go_end_bt_press(self, widget):
        self.set_label_text("Processing...")
        self.do_gui_update()
        self.display_map(self.voronoi_game.last_day - 1, 2)

    def view_drop_down_changed(self, widget, value):
        day = widget.get_key()
        if 0 <= day < self.voronoi_game.last_day:
            self.set_label_text("Processing...")
            self.do_gui_update()
            self.display_map(day, 0)

    def update_table(self):
        for i in range(constants.no_of_players):
            self.set_label_text("{}".format(self.voronoi_game.player_score[self.curr_day][self.curr_state][i]),
                                (i * 3) + 2)

            if self.curr_state == 2:
                self.set_label_text("{}".format(self.voronoi_game.player_total_score[self.curr_day][i]), (i * 3) + 3)
            elif self.curr_day == 0:
                self.set_label_text("{}".format(0), (i * 3) + 3)
            else:
                self.set_label_text("{}".format(self.voronoi_game.player_total_score[self.curr_day - 1][i]),
                                    (i * 3) + 3)
