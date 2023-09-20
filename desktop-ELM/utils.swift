//
//  utils.swift
//  desktop-ELM
//
//  Created by Anton Akusok on 29/07/2018.
//  Copyright © 2018 Anton Akusok. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

//let fp32stride = MemoryLayout<Float>.stride
//
//func MPSZeros(_ n: Int) -> MPSMatrix {
//    let descr = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * fp32stride, dataType: .float32)
//    let buffer = device.makeBuffer(length: n * n * fp32stride, options: [.storageModePrivate])!
//    return MPSMatrix(buffer: buffer, descriptor: descr)
//}
//
//func MPSZeros(rows: Int, columns: Int) -> MPSMatrix {
//    let descr = MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: columns * fp32stride, dataType: .float32)
//    let buffer = device.makeBuffer(length: rows * columns * fp32stride, options: [.storageModePrivate])!
//    return MPSMatrix(buffer: buffer, descriptor: descr)
//}
//
//func MPSEye(_ n: Int, alpha: Float) -> MPSMatrix {
//    let ptr = UnsafeMutablePointer<Float>.allocate(capacity: n*n)
//    ptr.initialize(repeating: 0.0, count: n*n)
//    for i in stride(from: 0, to: n*n, by: n+1) { ptr[i] = alpha }  // fill diagonal
//    
//    let descr = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * fp32stride, dataType: .float32)
//    let buffer = device.makeBuffer(bytes: ptr, length: n * n * fp32stride, options: [.storageModeManaged])!
//    ptr.deallocate()
//    return MPSMatrix(buffer: buffer, descriptor: descr)
//}

//func compute1(M: MPSMatrix) {
//    let cq = device.makeCommandQueue()!
//    var cbuf: MTLCommandBuffer
//    var blit: MTLBlitCommandEncoder
//
//    let vbuf = device.makeBuffer(bytes: [Float] (repeating: 1.0, count: M.columns), length: M.columns * MemoryLayout<Float>.stride, options: [])!
//    let V = MPSVector(buffer: vbuf, descriptor: MPSVectorDescriptor(length: M.columns, dataType: .float32))
//
//    let V2 = MPSVector(device: device, descriptor: MPSVectorDescriptor(length: M.rows, dataType: .float32))
//    let MV = MPSMatrixVectorMultiplication(device: device, rows: M.rows, columns: M.columns)
//
//    cbuf = cq.makeCommandBuffer()!
//    MV.encode(commandBuffer: cbuf, inputMatrix: M, inputVector: V, resultVector: V2)
//    cbuf.commit()
//
//    cbuf = cq.makeCommandBuffer()!
//    blit = cbuf.makeBlitCommandEncoder()!
//    blit.synchronize(resource: V2.data)
//    blit.endEncoding()
//    cbuf.commit()
//
//    cbuf.waitUntilCompleted()
//
//    let result = V2.data.contents().bindMemory(to: Float.self, capacity: M.rows)
//    print(Array(0..<10).map { result[$0] })
//}
