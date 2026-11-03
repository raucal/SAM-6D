import numpy as np
import os
import cv2

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def draw_3d_bbox(img, imgpts, color, size=3):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, size)

    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, size)

    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, size)
    return img

def draw_3d_pts(img, imgpts, color, size=1):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    for point in imgpts:
        img = cv2.circle(img, (point[0], point[1]), size, color, -1)
    return img

def draw_pose_axes(image, rotation, translation, intrinsics, length, axes_transform=None):
    """Draw RGB X/Y/Z axes; coordinates and translation must use the same units."""
    if not np.isfinite(length) or length<=0:return image
    axes=np.array([[0,length,0,0],[0,0,length,0],[0,0,0,length]],dtype=np.float64)
    if axes_transform is not None:
        transform=np.asarray(axes_transform,dtype=np.float64)
        if transform.shape!=(4,4) or not np.isfinite(transform).all():raise ValueError('Invalid axes transform')
        axes=transform[:3,:3]@axes+transform[:3,3,None]
    camera=np.asarray(rotation)@axes+np.asarray(translation).reshape(3,1)
    if not np.isfinite(camera).all() or np.any(camera[2]<=1e-6):return image
    uv=np.asarray(intrinsics)@camera;uv=(uv[:2]/uv[2]).T
    if not np.isfinite(uv).all():return image
    uv=np.rint(np.clip(uv,-1000000,1000000)).astype(np.int32)
    origin=tuple(uv[0])
    for index,(color,label) in enumerate([((255,0,0),'X'),((0,255,0),'Y'),((0,0,255),'Z')],1):
        endpoint=tuple(uv[index]);cv2.arrowedLine(image,origin,endpoint,color,3,tipLength=.12)
        cv2.putText(image,label,(endpoint[0]+4,endpoint[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    return image


def draw_detections(image, pred_rots, pred_trans, model_points, intrinsics, color=(255,0,0),
                    model_bounds=None, axes_transform=None, show_axes=True):
    draw_image_bbox=image.copy()
    points=np.asarray(model_points)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points):raise ValueError('Expected nonempty model points Nx3')
    bounds=np.asarray(model_bounds) if model_bounds is not None else np.stack([points.min(axis=0),points.max(axis=0)])
    if bounds.shape!=(2,3) or not np.isfinite(bounds).all():raise ValueError('Invalid model bounds')
    scale=bounds[1]-bounds[0]
    # The geometric center is independent of random surface sampling density.
    shift=(bounds[0]+bounds[1])/2
    bbox_3d=get_3d_bbox(scale,shift)
    choose=np.random.default_rng(0).choice(len(points),min(512,len(points)),replace=False)
    pts_3d=points[choose].T
    for ind in range(len(pred_rots)):
        transformed_bbox=pred_rots[ind]@bbox_3d+pred_trans[ind][:,None]
        if np.isfinite(transformed_bbox).all() and np.all(transformed_bbox[2]>1e-6):
            draw_image_bbox=draw_3d_bbox(draw_image_bbox,calculate_2d_projections(transformed_bbox,intrinsics[ind]),color)
        transformed_points=pred_rots[ind]@pts_3d+pred_trans[ind][:,None]
        valid=np.isfinite(transformed_points).all(axis=0)&(transformed_points[2]>1e-6)
        if valid.any():draw_image_bbox=draw_3d_pts(draw_image_bbox,calculate_2d_projections(transformed_points[:,valid],intrinsics[ind]),color)
        if show_axes:
            transform=axes_transform[ind] if axes_transform is not None and np.asarray(axes_transform).ndim==3 else axes_transform
            draw_pose_axes(draw_image_bbox,pred_rots[ind],pred_trans[ind],intrinsics[ind],.35*float(scale.max()),transform)
    return draw_image_bbox
