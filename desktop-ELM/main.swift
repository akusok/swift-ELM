//
//  main.swift
//  desktop-ELM
//
//  Created by Anton Akusok on 29/07/2018.
//  Copyright © 2018 Anton Akusok. All rights reserved.
//

import Foundation
import Accelerate
import MetalPerformanceShaders

print("Hello, World!")


let device = MTLCreateSystemDefaultDevice()!
guard MPSSupportsMTLDevice(device) else {
    fatalError("Error: This device has no Metal Performance Shaders")
}
let commandQueue = device.makeCommandQueue()!


// aux functions
let fp32stride = MemoryLayout<Float>.stride

func loadNpyToMatrix(contentsOf url: URL) -> MPSMatrix {
    let npy = try! Npy(contentsOf: url)
    let matBuffer = device.makeBuffer(length: npy.rows * npy.columns * fp32stride, options: [.storageModePrivate])!
    
    do {
        let dataBuffer = npy.elementsData.withUnsafeBytes {
            device.makeBuffer(bytes: $0, length: npy.count * npy.stride, options: [.storageModeShared])!
        }
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let blit = commandBuffer.makeBlitCommandEncoder()!
        blit.copy(from: dataBuffer, sourceOffset: 0, to: matBuffer, destinationOffset: 0, size: npy.count * fp32stride)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    let matDescriptor = MPSMatrixDescriptor(rows: npy.rows, columns: npy.columns, rowBytes: npy.columns * fp32stride, dataType: npy.mpsDataType)
    return MPSMatrix(buffer: matBuffer, descriptor: matDescriptor)
}

func loadNpyToVector(contentsOf url: URL) -> MPSVector {
    let npy = try! Npy(contentsOf: url)
    let vectBuffer = device.makeBuffer(length: npy.count * fp32stride, options: [.storageModePrivate])!
    
    do {
        let dataBuffer = npy.elementsData.withUnsafeBytes {
            device.makeBuffer(bytes: $0, length: npy.count * npy.stride, options: [.storageModeShared])!
        }
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let blit = commandBuffer.makeBlitCommandEncoder()!
        blit.copy(from: dataBuffer, sourceOffset: 0, to: vectBuffer, destinationOffset: 0, size: npy.count * fp32stride)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    let vectDescriptor = MPSVectorDescriptor(length: npy.columns, dataType: npy.mpsDataType)
    return MPSVector(buffer: vectBuffer, descriptor: vectDescriptor)
}

func MPSZeros(_ n: Int) -> MPSMatrix {
    let descr = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * fp32stride, dataType: .float32)
    let buffer = device.makeBuffer(length: n * n * fp32stride, options: [.storageModePrivate])!
    return MPSMatrix(buffer: buffer, descriptor: descr)
}

func MPSZeros(rows: Int, columns: Int) -> MPSMatrix {
    let descr = MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: columns * fp32stride, dataType: .float32)
    let buffer = device.makeBuffer(length: rows * columns * fp32stride, options: [.storageModePrivate])!
    return MPSMatrix(buffer: buffer, descriptor: descr)
}

func MPSEye(_ n: Int, alpha: Float) -> MPSMatrix {
    let descr = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * fp32stride, dataType: .float32)
    let buffer = device.makeBuffer(length: n*n*fp32stride, options: [.storageModePrivate])!
    let matrix = MPSMatrix(buffer: buffer, descriptor: descr)
    
    // generate and upload data to private matrix
    do {
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: n*n)
        defer { ptr.deallocate() }
        
        ptr.initialize(repeating: 0.0, count: n*n)
        for i in stride(from: 0, to: n*n, by: n+1) { ptr[i] = alpha }  // fill diagonal
        let localBuffer = device.makeBuffer(bytes: ptr, length: n*n*fp32stride, options: [])!
        
        let cbuf = commandQueue.makeCommandBuffer()!
        let blit = cbuf.makeBlitCommandEncoder()!
        blit.copy(from: localBuffer, sourceOffset: 0, to: buffer, destinationOffset: 0, size: n*n*fp32stride)
        blit.endEncoding()
        cbuf.commit()
        cbuf.waitUntilCompleted()
    }
    
    return matrix
}

func sample(M: MPSMatrix) {
    let tempBuff = device.makeBuffer(length: 10 * fp32stride, options: [])!
    
    let cbuf = commandQueue.makeCommandBuffer()!
    let blit = cbuf.makeBlitCommandEncoder()!
    blit.copy(from: M.data, sourceOffset: 0, to: tempBuff, destinationOffset: 0, size: 10 * fp32stride)
    blit.endEncoding()
    cbuf.commit()
    cbuf.waitUntilCompleted()
    
    let result = tempBuff.contents().assumingMemoryBound(to: Float.self)
    print(Array(0..<10).map { result[$0] })
}





// ####################################################################################################################

var times = Array<Double>()

//for bsize in stride(from: 104, to: 300, by: 8) {
do {
    let t0 = CFAbsoluteTimeGetCurrent()

    // set constants
    let inputs: Int = 7*7*3
    let targets: Int = 2
    let numSamples: Int = 2  // 100
    let useRealFiles = true
    let saveResults = true
    let bsize = 56

    let bK = 2
    let bL = 1024
    let alpha: Float = 0.5
    let dataB: Float = 0.0
    let dataStd: Float = 64.0 * powf(Float(inputs) / Float(3.0), 0.5)
    let dataBiasArray: [Float]? = [122.5, 108.4,  99.9, 122.5, 108.4,  99.9, 122.5, 108.4,  99.9,
                                   122.5, 108.4,  99.9, 122.5, 108.4,  99.9, 122.5, 108.4,  99.9,
                                   122.5, 108.4,  99.9, 122.6, 108.4,  99.9, 122.6, 108.4,  99.9,
                                   122.6, 108.4,  99.9, 122.5, 108.4,  99.9, 122.6, 108.4,  99.9,
                                   122.6, 108.4,  99.9, 122.6, 108.4,  99.9, 122.6, 108.4,  99.9,
                                   122.6, 108.3,  99.9, 122.6, 108.4,  99.9, 122.6, 108.3,  99.9,
                                   122.6, 108.3,  99.9, 122.6, 108.3,  99.9, 122.6, 108.3,  99.9,
                                   122.6, 108.3,  99.9, 122.6, 108.3,  99.9, 122.6, 108.3,  99.9,
                                   122.6, 108.3,  99.9, 122.6, 108.3,  99.9, 122.6, 108.3,  99.9,
                                   122.6, 108.3,  99.9, 122.6, 108.3,  99.9, 122.6, 108.3,  99.9,
                                   122.6, 108.3,  99.9, 122.6, 108.3,  99.9, 122.6, 108.3,  99.9,
                                   122.6, 108.3,  99.9, 122.6, 108.3,  99.9, 122.6, 108.3,  99.9,
                                   122.6, 108.3,  99.9, 122.6, 108.3,  99.9, 122.6, 108.3,  99.9,
                                   122.6, 108.3,  99.9, 122.6, 108.3,  99.9, 122.6, 108.3,  99.9,
                                   122.6, 108.3,  99.9, 122.6, 108.3,  99.9, 122.6, 108.3,  99.9,
                                   122.6, 108.3,  99.9, 122.6, 108.3,  99.9, 122.6, 108.3,  99.9,
                                   122.6, 108.3,  99.9]
    
    

    
    // load data
    let filesW = Array(0 ..< bK).map { i in URL(fileURLWithPath: "/Users/akusok/GitHub/swift-ELM/data/w_\(i).npy") }
    let filesBias = Array(0 ..< bK).map { i in URL(fileURLWithPath: "/Users/akusok/GitHub/swift-ELM/data/bias_\(i).npy") }

    var X: MPSMatrix
    var Y: MPSMatrix
    
    // prepare MPS storage
    let W = filesW.map { loadNpyToMatrix(contentsOf: $0) }
    let bias = filesBias.map { loadNpyToVector(contentsOf: $0) }

    var dataBiasVector: MPSVector? = nil
    if let dataBiasArray = dataBiasArray {
        let dataBiasBuffer = device.makeBuffer(bytes: dataBiasArray.map { Float(-1.0) * $0 }, length: dataBiasArray.count * fp32stride, options: [.storageModeManaged])!
        dataBiasVector = MPSVector(buffer: dataBiasBuffer, descriptor: MPSVectorDescriptor(length: dataBiasArray.count, dataType: .float32))
    }

    let rowBytesX = MPSMatrixDescriptor.rowBytes(fromColumns: inputs, dataType: .float32)
    let bufferX = device.makeBuffer(length: bsize * rowBytesX, options: [.storageModePrivate])!
    let batchX = MPSMatrix(buffer: bufferX, descriptor: MPSMatrixDescriptor(rows: bsize, columns: inputs, rowBytes: rowBytesX, dataType: .float32))

    let rowBytesY = MPSMatrixDescriptor.rowBytes(fromColumns: targets, dataType: .float32)
    let bufferY = device.makeBuffer(length: bsize * rowBytesY, options: [.storageModePrivate])!
    let batchY = MPSMatrix(buffer: bufferY, descriptor: MPSMatrixDescriptor(rows: bsize, columns: targets, rowBytes: rowBytesY, dataType: .float32))
    
    let H = Array(0 ..< bK).map { _ in MPSZeros(rows: bsize, columns: bL) }
    let B = Array(0 ..< bK).map { _ in MPSZeros(rows: bL, columns: targets) }
    var L: [[MPSMatrix?]] = Array(repeating: Array(repeating: nil, count: bK), count: bK)
    for row in 0 ..< bK {
        L[row][row] = MPSEye(bL, alpha: alpha)
        for col in 0 ..< row {
            L[row][col] = MPSZeros(bL)
        }
    }

    
    // BATCH UPDATE

    // prepare kernels
    let normalizeBatch = MPSMatrixNeuron(device: device)
    normalizeBatch.setNeuronType(.linear, parameterA: 1.0/dataStd, parameterB: -dataB, parameterC: 0.0)
    
    let matMulXW = MPSMatrixMultiplication(device: device, resultRows: bsize, resultColumns: bL, interiorColumns: inputs)
    let matMulHtH = MPSMatrixMultiplication(device: device, transposeLeft: true, transposeRight: false, resultRows: bL, resultColumns: bL, interiorColumns: bsize, alpha: 1.0, beta: 1.0)
    let matMulHtT = MPSMatrixMultiplication(device: device, transposeLeft: true, transposeRight: false, resultRows: bL, resultColumns: targets, interiorColumns: bsize, alpha: 1.0, beta: 1.0)
    let matBiasTanh = MPSMatrixNeuron(device: device)
    matBiasTanh.setNeuronType(.tanH, parameterA: 1.0, parameterB: 1.0, parameterC: 0.0)
    
    for b in 1...numSamples {
        
        // load data
        if useRealFiles {
            let fileX = URL(fileURLWithPath: "/Users/akusok/GitHub/swift-ELM/data/X_\(b).npy")
            let fileY = URL(fileURLWithPath: "/Users/akusok/GitHub/swift-ELM/data/Y_\(b).npy")
            X = loadNpyToMatrix(contentsOf: fileX)
            Y = loadNpyToMatrix(contentsOf: fileY)
        } else {
            let descrX = MPSMatrixDescriptor(rows: 123456, columns: inputs, rowBytes: inputs*MemoryLayout<Float>.stride, dataType: .float32)
            let descrY = MPSMatrixDescriptor(rows: 123456, columns: 2, rowBytes: 2*MemoryLayout<Float>.stride, dataType: .float32)
            X = MPSMatrix(device: device, descriptor: descrX)
            Y = MPSMatrix(device: device, descriptor: descrY)
        }
        
        print(">>> Batch \(b)")
        
        for offset in stride(from: 0, to: X.rows, by: bsize) {
            
            let batchRows = min(X.rows - offset, bsize)
            //print("Processing \(offset)+\(batchRows) from total \(X.rows) rows of data.")

            // prepare batch of data
            let copyOffset = MPSMatrixCopyOffsets(sourceRowOffset: UInt32(offset), sourceColumnOffset: 0, destinationRowOffset: 0, destinationColumnOffset: 0)
            let uploadX = MPSMatrixCopy(device: device, copyRows: batchRows, copyColumns: inputs, sourcesAreTransposed: false, destinationsAreTransposed: false)
            let uploadY = MPSMatrixCopy(device: device, copyRows: batchRows, copyColumns: targets, sourcesAreTransposed: false, destinationsAreTransposed: false)

            let uploadBuf = commandQueue.makeCommandBuffer()!
            uploadX.encode(commandBuffer: uploadBuf, copyDescriptor: MPSMatrixCopyDescriptor(sourceMatrix: X, destinationMatrix: batchX, offsets: copyOffset))
            uploadY.encode(commandBuffer: uploadBuf, copyDescriptor: MPSMatrixCopyDescriptor(sourceMatrix: Y, destinationMatrix: batchY, offsets: copyOffset))
            uploadBuf.commit()

            //  normalize uploaded data
            let normBuf = commandQueue.makeCommandBuffer()!
            normalizeBatch.encode(commandBuffer: normBuf, inputMatrix: batchX, biasVector: dataBiasVector, resultMatrix: batchX)
            normBuf.commit()

            // load batch data to GPU (normalizing on the way)
            if batchRows < bsize {  // reset batchX to zeros for incomplete batch of X
                let cropBuf = commandQueue.makeCommandBuffer()!
                let blit = cropBuf.makeBlitCommandEncoder()!
                blit.fill(buffer: batchX.data, range: batchX.rowBytes*batchRows ..< batchX.matrixBytes, value: 0)
                blit.endEncoding()
                cropBuf.commit()
            }
            
            // encode and run computations
            let bufXW = commandQueue.makeCommandBuffer()!
            for i in 0 ..< bK {
                matMulXW.encode(commandBuffer: bufXW, leftMatrix: batchX, rightMatrix: W[i], resultMatrix: H[i])
            }
            bufXW.commit()
            
            let bufBiasTanh = commandQueue.makeCommandBuffer()!
            for i in 0 ..< bK {
                matBiasTanh.encode(commandBuffer: bufBiasTanh, inputMatrix: H[i], biasVector: bias[i], resultMatrix: H[i])
            }
            bufBiasTanh.commit()
            
            let bufLB = commandQueue.makeCommandBuffer()!  // .makeCommandBufferWithUnretainedReferences()!
            for i in 0 ..< bK {
                matMulHtT.encode(commandBuffer: bufLB, leftMatrix: H[i], rightMatrix: batchY, resultMatrix: B[i])
                for j in 0 ... i {
                    matMulHtH.encode(commandBuffer: bufLB, leftMatrix: H[i], rightMatrix: H[j], resultMatrix: L[i][j]!)
                }
            }
            bufLB.commit()
        }
    }
    
    
    // SOLVE
    
    // setup kernels
    let runCho = MPSMatrixDecompositionCholesky(device: device, lower: true, order: bL)
    let updateL1 = MPSMatrixSolveTriangular(device: device, right: true, upper: false, transpose: true, unit: false, order: bL, numberOfRightHandSides: bL, alpha: 1.0)
    let updateL2 = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true, resultRows: bL, resultColumns: bL, interiorColumns: bL, alpha: -1.0, beta: 1.0)
    
    // batch Cholesky solver
    for i in 0 ..< bK {
        let bufCho = commandQueue.makeCommandBuffer()!
        runCho.encode(commandBuffer: bufCho, sourceMatrix: L[i][i]!, resultMatrix: L[i][i]!, status: nil)
        bufCho.commit()
        
        let bufUpdateL1 = commandQueue.makeCommandBuffer()!
        for j in i+1 ..< bK {
            updateL1.encode(commandBuffer: bufUpdateL1, sourceMatrix: L[i][i]!, rightHandSideMatrix: L[j][i]!, solutionMatrix: L[j][i]!)
        }
        bufUpdateL1.commit()
        
        let bufUpdateL2 = commandQueue.makeCommandBuffer()!
        for j in i+1 ..< bK {
            for k in j ..< bK {
                updateL2.encode(commandBuffer: bufUpdateL2, leftMatrix: L[k][i]!, rightMatrix: L[j][i]!, resultMatrix: L[k][j]!)
            }
        }
        bufUpdateL2.commit()
    }
    
    // setup kernels
    let solveTri1 = MPSMatrixSolveTriangular(device: device, right: false, upper: false, transpose: false, unit: false, order: bL, numberOfRightHandSides: targets, alpha: 1.0)
    let solveTri2 = MPSMatrixSolveTriangular(device: device, right: false, upper: false, transpose: true, unit: false, order: bL, numberOfRightHandSides: targets, alpha: 1.0)
    let updateT = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: bL, resultColumns: targets, interiorColumns: bL, alpha: -1.0, beta: 1.0)
    let updateT2 = MPSMatrixMultiplication(device: device, transposeLeft: true, transposeRight: false, resultRows: bL, resultColumns: targets, interiorColumns: bL, alpha: -1.0, beta: 1.0)
    
    // forward substitution
    for i in 0 ..< bK {
        let bufSolveTri1 = commandQueue.makeCommandBuffer()!
        solveTri1.encode(commandBuffer: bufSolveTri1, sourceMatrix: L[i][i]!, rightHandSideMatrix: B[i], solutionMatrix: B[i])
        bufSolveTri1.commit()
        
        let bufSolveUpdate1 = commandQueue.makeCommandBuffer()!
        for j in i+1 ..< bK {
            updateT.encode(commandBuffer: bufSolveUpdate1, leftMatrix: L[j][i]!, rightMatrix: B[i], resultMatrix: B[j])
        }
        bufSolveUpdate1.commit()
    }
    
    // backward substitution
    for i in (0 ..< bK).reversed() {
        let bufSolveTri2 = commandQueue.makeCommandBuffer()!
        solveTri2.encode(commandBuffer: bufSolveTri2, sourceMatrix: L[i][i]!, rightHandSideMatrix: B[i], solutionMatrix: B[i])
        bufSolveTri2.commit()
        
        let bufSolveUpdate2 = commandQueue.makeCommandBuffer()!
        for j in 0 ..< i {
            updateT2.encode(commandBuffer: bufSolveUpdate2, leftMatrix: L[i][j]!, rightMatrix: B[i], resultMatrix: B[j])
        }
        bufSolveUpdate2.commit()
    }
    
    L = Array(repeating: Array(repeating: nil, count: bK), count: bK)
    
    // wait uptil everything completes
    let cbuf = commandQueue.makeCommandBuffer()!
    cbuf.commit()
    cbuf.waitUntilCompleted()
    
    if saveResults {
        for (n, b1) in B.enumerated() {
            let localBuffer = device.makeBuffer(length: b1.matrixBytes, options: [.storageModeShared])!
            let cbuf = commandQueue.makeCommandBuffer()!
            let blit = cbuf.makeBlitCommandEncoder()!
            blit.copy(from: b1.data, sourceOffset: 0, to: localBuffer, destinationOffset: 0, size: b1.matrixBytes)
            blit.endEncoding()
            cbuf.commit()
            cbuf.waitUntilCompleted()

            let result = UnsafeMutableBufferPointer(start: localBuffer.contents().bindMemory(to: Float.self, capacity: b1.rows * b1.columns),
                                                    count: b1.rows * b1.columns)
            let myData = Data(buffer: result)

            let npy = Npy(elementsData: myData, shape: [b1.rows, b1.columns], dataType: .float32, endian: .host, isFortranOrder: false, descr: "'<f4'")
//            try! npy.save(to: URL(fileURLWithPath: "/Users/akusok/wrkdir/b_\(n).npy"))
        }
    }

    let t = CFAbsoluteTimeGetCurrent() - t0
    print("Solution with bsize=\(bsize) took \(t) seconds")
    times.append(t)
}

// ####################################################################################################################

//let res = Yh.data.contents().bindMemory(to: Float.self, capacity: Yh.rows * c)
//for i in 0 ..< 10 {
//    print(Array(0 ..< c).map { res[c*i + $0] })
//}

//print(Array(stride(from: 104, to: 300, by: 8)))
//print(times)

print("Done here!")
sleep(2000)


