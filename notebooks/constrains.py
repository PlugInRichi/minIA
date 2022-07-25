HOLA =1

DEBIASED = [
    'dr7objid',
    'sample',
    'total_classifications',
    'total_votes',
    't01_smooth_or_features_a01_smooth_debiased',
    't01_smooth_or_features_a02_features_or_disk_debiased',
    't01_smooth_or_features_a03_star_or_artifact_debiased',
    't02_edgeon_a04_yes_debiased',
    't02_edgeon_a05_no_debiased',
    't03_bar_a06_bar_debiased',
    't03_bar_a07_no_bar_debiased',
    't04_spiral_a08_spiral_debiased',
    't04_spiral_a09_no_spiral_debiased',
    't05_bulge_prominence_a10_no_bulge_debiased',
    't05_bulge_prominence_a11_just_noticeable_debiased',
    't05_bulge_prominence_a12_obvious_debiased',
    't05_bulge_prominence_a13_dominant_debiased',
    't06_odd_a14_yes_debiased',
    't06_odd_a15_no_debiased',
    't07_rounded_a16_completely_round_debiased',
    't07_rounded_a17_in_between_debiased',
    't07_rounded_a18_cigar_shaped_debiased',
    't08_odd_feature_a19_ring_debiased',
    't08_odd_feature_a20_lens_or_arc_debiased',
    't08_odd_feature_a21_disturbed_debiased',
    't08_odd_feature_a22_irregular_debiased',
    't08_odd_feature_a23_other_debiased',
    't08_odd_feature_a24_merger_debiased',
    't08_odd_feature_a38_dust_lane_debiased',
    't09_bulge_shape_a25_rounded_debiased',
    't09_bulge_shape_a26_boxy_debiased',
    't09_bulge_shape_a27_no_bulge_debiased',
    't10_arms_winding_a28_tight_debiased',
    't10_arms_winding_a29_medium_debiased',
    't10_arms_winding_a30_loose_debiased',
    't11_arms_number_a31_1_debiased',
    't11_arms_number_a32_2_debiased',
    't11_arms_number_a33_3_debiased',
    't11_arms_number_a34_4_debiased',
    't11_arms_number_a36_more_than_4_debiased',
    't11_arms_number_a37_cant_tell_debiased'
]

CLASS_NAMES = [
    't08_ring',
    't08_lens_or_arc',
    't08_disturbed',
    't08_irregular',
    't08_merger',
    't08_dust_lane',

    't11_arms_number_1',
    't11_arms_number_2',
    #'t11_arms_number_3',
    #'t11_arms_number_4',

    't05_no_bulge',
    't05_just_noticeable',
    't05_obvious',
    't05_dominant',

    't07_completely_round',
    't07_in_between',
    't07_cigar_shaped',

    't09_rounded',
    't09_boxy',
    't09_no_bulge',

    't10_tight',
    't10_medium',
    't10_loose']

SELECTED_TYPES = [
    'dr7objid',
    'total_votes',
    't01_smooth_or_features_a01_smooth_debiased',
    't01_smooth_or_features_a02_features_or_disk_debiased',
    't05_bulge_prominence_a10_no_bulge_debiased',
    't05_bulge_prominence_a11_just_noticeable_debiased',
    't05_bulge_prominence_a12_obvious_debiased',
    't05_bulge_prominence_a13_dominant_debiased',
    't07_rounded_a16_completely_round_debiased',
    't07_rounded_a17_in_between_debiased',
    't07_rounded_a18_cigar_shaped_debiased',
    't08_odd_feature_a19_ring_debiased',
    't08_odd_feature_a20_lens_or_arc_debiased',
    't08_odd_feature_a21_disturbed_debiased',
    't08_odd_feature_a22_irregular_debiased',
    't08_odd_feature_a24_merger_debiased',
    't08_odd_feature_a38_dust_lane_debiased',
    't09_bulge_shape_a25_rounded_debiased',
    't09_bulge_shape_a26_boxy_debiased',
    't09_bulge_shape_a27_no_bulge_debiased',
    't10_arms_winding_a28_tight_debiased',
    't10_arms_winding_a29_medium_debiased',
    't10_arms_winding_a30_loose_debiased',
    't11_arms_number_a31_1_debiased',
    't11_arms_number_a32_2_debiased',
    't11_arms_number_a33_3_debiased',
    't11_arms_number_a34_4_debiased'
]

ALL_TYPES = [
    't01_smooth_or_features_a01_smooth_debiased',
    't01_smooth_or_features_a02_features_or_disk_debiased',
    't01_smooth_or_features_a03_star_or_artifact_debiased',
    't02_edgeon_a04_yes_debiased',
    't02_edgeon_a05_no_debiased',
    't03_bar_a06_bar_debiased',
    't03_bar_a07_no_bar_debiased',
    't04_spiral_a08_spiral_debiased',
    't04_spiral_a09_no_spiral_debiased',
    't05_bulge_prominence_a10_no_bulge_debiased',
    't05_bulge_prominence_a11_just_noticeable_debiased',
    't05_bulge_prominence_a12_obvious_debiased',
    't05_bulge_prominence_a13_dominant_debiased',
    't06_odd_a14_yes_debiased',
    't06_odd_a15_no_debiased',
    't07_rounded_a16_completely_round_debiased',
    't07_rounded_a17_in_between_debiased',
    't07_rounded_a18_cigar_shaped_debiased',
    't08_odd_feature_a19_ring_debiased',
    't08_odd_feature_a20_lens_or_arc_debiased',
    't08_odd_feature_a21_disturbed_debiased',
    't08_odd_feature_a22_irregular_debiased',
    't08_odd_feature_a23_other_debiased',
    't08_odd_feature_a24_merger_debiased',
    't08_odd_feature_a38_dust_lane_debiased',
    't09_bulge_shape_a25_rounded_debiased',
    't09_bulge_shape_a26_boxy_debiased',
    't09_bulge_shape_a27_no_bulge_debiased',
    't10_arms_winding_a28_tight_debiased',
    't10_arms_winding_a29_medium_debiased',
    't10_arms_winding_a30_loose_debiased',
    't11_arms_number_a31_1_debiased',
    't11_arms_number_a32_2_debiased',
    't11_arms_number_a33_3_debiased',
    't11_arms_number_a34_4_debiased',
    't11_arms_number_a36_more_than_4_debiased',
    't11_arms_number_a37_cant_tell_debiased'
]