struct box {
    float2 tl;
    float2 tr;
    float2 br;
    float2 bl;
};

struct line {
    float2 a;
    float2 b;
};

// Cross product but only returns magnitude because Z is always out of screen
float cross2(float2 a, float2 b) {
    return a[0] * b[1] - a[1] * b[0];
}

float2 line_line_intersection(struct line l1, struct line l2) {
    // We came up with some complicated way of doing it by hand, but just
    // looked up the formula later. Sourced from chatgpt :( so no nice link
    // to give out

    float2 a = l1.a;
    float2 b = l1.b;
    float2 c = l2.a;
    float2 d = l2.b;

    float2 r = b - a;
    float2 s = d - c;

    float denom = cross2(r, s);
    if (denom == 0) return (float2){INFINITY, INFINITY};

    float t = cross2(c - a, s) / denom;
    return a + r * t;
}

struct box box_from_data_repr(__global float* data) {
    // Takes cx,cy,w,h,r
    // Returns corner points
    float half_w = data[2] / 2.0f;
    float half_h = data[3] / 2.0f;
    float r = data[4];

    float2 center = {data[0], data[1]};
    float2 x_axis = {cos(r), sin(r)};
    float2 y_axis = {-x_axis[1], x_axis[0]};

    float2 x_edge_offs = (half_w * x_axis);
    float2 y_edge_offs = (half_h * y_axis);

    // Consider top left 0, 0
    float2 tl = center - x_edge_offs - y_edge_offs;
    float2 tr = center + x_edge_offs - y_edge_offs;
    float2 br = center + x_edge_offs + y_edge_offs;
    float2 bl = center - x_edge_offs + y_edge_offs;

    return (struct box) {
        .tl = tl,
        .tr = tr,
        .bl = bl,
        .br = br,
    };
}

float box_area_from_data_repr(__global float* data) {
    return data[2] * data[3];
}

// Growable array of points, up to 8 which is the max for 2
// overlapping rectangles
struct intersection_polygon {
    float2 data[8];
    unsigned char len;
};

void intersection_polygon_init(struct intersection_polygon* builder) {
    builder->len = 0;
}

void polygon_builder_push(struct intersection_polygon* builder, float2 val) {
    builder->data[builder->len] = val;
    builder->len++;
}

// If p is on one side of the line, clip, otherwise do not clip
bool should_clip_vertex(struct line line, float2 p) {
    float2 ab = line.b - line.a;
    float2 rot_90_axis = normalize((float2){ ab[1], -ab[0] });

    float2 ap = p - line.a;
    return dot(ap, rot_90_axis) >= 0;
}

bool point_in_bounds(float2 point, struct line line) {
    float2 ap = point - line.a;
    float2 ab = line.b - line.a;
    float ab_len = length(ab);
    float2 ab_norm = ab / ab_len;
    float d = dot(ap, ab_norm);
    return d <= ab_len && d >= 0;
}

struct intersection_polygon calc_intersection(struct box b1, struct box b2) {
    // Algorithm stolen from
    // https://www.wikiwand.com/en/articles/Sutherland%E2%80%93Hodgman_algorithm
    //
    // * Iterate lines on one polygon
    // * Remove points outside
    // * Add points that intersect with clip line
    // * Iterate all lines

    // Double buffer for iterations. Each iter takes output from last, so we
    // need space for both
    struct intersection_polygon builders[2];

    intersection_polygon_init(&builders[0]);
    polygon_builder_push(&builders[0], b2.tl);
    polygon_builder_push(&builders[0], b2.tr);
    polygon_builder_push(&builders[0], b2.br);
    polygon_builder_push(&builders[0], b2.bl);

    unsigned char out_idx = 0;
    float2 clip_points[4] = {b1.tl, b1.tr, b1.br, b1.bl};

    for (int clip_idx = 0; clip_idx < 4; clip_idx++) {
        struct intersection_polygon *in = &builders[out_idx];

        out_idx = out_idx ^ 1;
        struct intersection_polygon *out = &builders[out_idx];

        intersection_polygon_init(out);

        struct line clip_edge = {
            clip_points[clip_idx],
            clip_points[(clip_idx + 1) % 4]
        };

        for (unsigned char i = 0; i < in->len; i++ ){
            float2 p = in->data[i];
            if (!should_clip_vertex(clip_edge, p)) {
                polygon_builder_push(out, p);
            }

            float2 next_p = in->data[(i + 1) % in->len];
            struct line intersection_line = { p, next_p };
            float2 intersection_point = line_line_intersection(clip_edge, intersection_line);
            if (point_in_bounds(intersection_point, intersection_line)) {
                polygon_builder_push(out, intersection_point);
            }
        }
    }

    return builders[out_idx];
}

float2 calc_poly_average(struct intersection_polygon poly) {
    float2 average = {0.0f, 0.0f};
    for (int i = 0; i < poly.len; i++) {
        average += poly.data[i];
    }
    return average / (float)poly.len;
}

struct poly_fan_iter {
    const struct intersection_polygon* edge_points;
    float2 center;
    int idx;
};

void poly_fan_iter_init(struct poly_fan_iter* it, struct intersection_polygon* poly) {
    float2 average_point = calc_poly_average(*poly);
    *it = (struct poly_fan_iter){
        poly,
        average_point,
        // Start < 0 so first call to next() initializes us
        -1,
    };
}

bool poly_fan_iter_next(struct poly_fan_iter* it) {
    it->idx += 1;
    if (it->idx >= it->edge_points->len) return false;
    return true;
}

struct poly_fan_tri {
    float2 a;
    float2 b;
    float2 c;
};

struct poly_fan_tri poly_fan_iter_triangle(struct poly_fan_iter* it) {
    return (struct poly_fan_tri) {
        it->center,
        it->edge_points->data[it->idx],
        it->edge_points->data[(it->idx + 1) % it->edge_points->len],
    };
}


float poly_fan_tri_area(struct poly_fan_tri tri) {
    // The first place I could find that explained this formula in a way that
    // made sense. I am 10 years old
    // https://www.math-only-math.com/area-of-the-triangle-formed-by-three-co-ordinate-points.html
    return 0.5f * (
            tri.a[0] * (tri.b[1] - tri.c[1]) +
            tri.b[0] * (tri.c[1] - tri.a[1]) +
            tri.c[0] * (tri.a[1] - tri.b[1])
        );
}

__kernel void calc_iou(
    __global float* as,
    __global float* bs,
    __global float* out,
    uint n
) {
    // box repr is cx, cy, w, h, rot_rad
    // as, bs (5, N)
    // out (n)

    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    __global float* in_a = as + 5 * global_id;
    __global float* in_b = bs + 5 * global_id;

    struct box a_box = box_from_data_repr(in_a);
    struct box b_box = box_from_data_repr(in_b);

    struct intersection_polygon poly = calc_intersection(a_box, b_box);

    struct poly_fan_iter it;
    poly_fan_iter_init(&it, &poly);

    float intersection_area = 0.0f;
    while (poly_fan_iter_next(&it)) {
        struct poly_fan_tri tri = poly_fan_iter_triangle(&it);
        intersection_area += poly_fan_tri_area(tri);
    }

    float union_area = box_area_from_data_repr(in_a) + box_area_from_data_repr(in_b) - intersection_area;
    out[global_id] = intersection_area / union_area;
}
